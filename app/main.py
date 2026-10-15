import calendar
import os
from contextlib import asynccontextmanager
from datetime import date as dt_date, datetime, time as dt_time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Annotated, Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv(override=False)

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

MIN_EPOCH_MS = 100_000_000_000
MAX_EPOCH_MS = 4_102_444_800_000

StatusType = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatusType = Literal["PRESENT", "WFH", "ON_DUTY"]
ExplainEndpointType = Literal[
    "attendance_list",
    "employee_monthly",
    "department_summary",
    "late_leaderboard",
    "department_trend",
]
PRESENCE_STATUSES = ("PRESENT", "WFH", "ON_DUTY")


def get_db():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGO_DB", "attendance_db")
    client = MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=5000)
    return client, client[db_name]


client: Optional[MongoClient] = None
db: Any = None


def ensure_indexes(database: Any) -> None:
    database.employees.create_index([("emp_code", ASCENDING)], unique=True, name="idx_emp_code_unique")
    database.employees.create_index(
        [("department", ASCENDING), ("joined_on", ASCENDING), ("emp_code", ASCENDING)],
        name="idx_dept_joined_emp",
    )
    database.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="idx_emp_date_unique",
    )
    database.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)],
        name="idx_date_desc_emp_asc",
    )
    database.attendance_logs.create_index(
        [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)],
        name="idx_status_date_emp",
    )
    database.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("punch_in", DESCENDING)],
        name="idx_emp_punch_in_desc",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db
    client, db = get_db()
    try:
        ensure_indexes(db)
    except PyMongoError:
        pass
    yield
    if client is not None:
        client.close()


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


def round_half_up(val: float | Decimal, decimals: int = 2) -> float:
    d = Decimal(str(val))
    target = Decimal("10") ** -decimals if decimals > 0 else Decimal("1")
    return float(d.quantize(target, rounding=ROUND_HALF_UP))


def ms_to_utc_dt(ms: int, truncate_seconds: bool = True) -> datetime:
    if truncate_seconds:
        ms = (ms // 1000) * 1000
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def dt_to_ms(dt_val: Optional[datetime]) -> Optional[int]:
    if dt_val is None:
        return None
    if dt_val.tzinfo is None:
        dt_val = dt_val.replace(tzinfo=UTC)
    return int(dt_val.timestamp() * 1000)


def now_utc_truncated() -> datetime:
    ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    return ms_to_utc_dt(ms, truncate_seconds=True)


def parse_hhmm(hhmm: str) -> tuple[int, int]:
    parts = hhmm.split(":")
    return int(parts[0]), int(parts[1])


def is_overnight_shift(shift_start: str, shift_end: str) -> bool:
    return shift_end <= shift_start


def compute_attendance_date(punch_in_utc: datetime, shift_start: str, shift_end: str) -> str:
    if punch_in_utc.tzinfo is None:
        punch_in_utc = punch_in_utc.replace(tzinfo=UTC)
    ist_dt = punch_in_utc.astimezone(IST)
    ist_date = ist_dt.date()
    if is_overnight_shift(shift_start, shift_end):
        eh, em = parse_hhmm(shift_end)
        if (ist_dt.hour, ist_dt.minute, ist_dt.second) < (eh, em, 0):
            ist_date = ist_date - timedelta(days=1)
    return ist_date.isoformat()


def get_shift_bounds_utc(att_date_str: str, shift_start: str, shift_end: str) -> tuple[datetime, datetime]:
    d = dt_date.fromisoformat(att_date_str)
    sh, sm = parse_hhmm(shift_start)
    eh, em = parse_hhmm(shift_end)
    start_ist = datetime.combine(d, dt_time(sh, sm, 0), tzinfo=IST)
    end_date = d + timedelta(days=1) if is_overnight_shift(shift_start, shift_end) else d
    end_ist = datetime.combine(end_date, dt_time(eh, em, 0), tzinfo=IST)
    return start_ist.astimezone(UTC), end_ist.astimezone(UTC)


def compute_late_minutes(punch_in_utc: datetime, att_date_str: str, shift_start: str, shift_end: str) -> int:
    start_utc, _ = get_shift_bounds_utc(att_date_str, shift_start, shift_end)
    if punch_in_utc.tzinfo is None:
        punch_in_utc = punch_in_utc.replace(tzinfo=UTC)
    diff_seconds = int((punch_in_utc - start_utc).total_seconds())
    if diff_seconds > 600:
        return diff_seconds // 60
    return 0


def compute_punch_out_metrics(
    punch_in_utc: datetime,
    punch_out_utc: datetime,
    att_date_str: str,
    shift_start: str,
    shift_end: str,
) -> tuple[float, int, bool]:
    if punch_in_utc.tzinfo is None:
        punch_in_utc = punch_in_utc.replace(tzinfo=UTC)
    if punch_out_utc.tzinfo is None:
        punch_out_utc = punch_out_utc.replace(tzinfo=UTC)

    elapsed_sec = int((punch_out_utc - punch_in_utc).total_seconds())
    work_hours = round_half_up(Decimal(elapsed_sec) / Decimal(3600), 2)

    _, end_utc = get_shift_bounds_utc(att_date_str, shift_start, shift_end)
    ot_seconds = int((punch_out_utc - end_utc).total_seconds())
    ot_minutes = ot_seconds // 60 if ot_seconds >= 1800 else 0

    half_day = work_hours < 4.50
    return work_hours, ot_minutes, half_day


def parse_strict_date(val: str, field_name: str = "date") -> dt_date:
    if not isinstance(val, str) or len(val) != 10:
        raise ValueError(f"Invalid {field_name}; expected YYYY-MM-DD")
    try:
        return dt_date.fromisoformat(val)
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date for {field_name}: {val}") from exc


def validate_month_str(month: str) -> tuple[int, int, str, str]:
    if not isinstance(month, str) or len(month) != 7 or month[4] != "-":
        raise HTTPException(status_code=422, detail=[{"msg": "Invalid month format; expected YYYY-MM"}])
    try:
        year = int(month[:4])
        m = int(month[5:7])
        if not (1 <= m <= 12):
            raise ValueError()
    except ValueError:
        raise HTTPException(status_code=422, detail=[{"msg": "Invalid month format; expected YYYY-MM"}])
    _, last_day = calendar.monthrange(year, m)
    start_str = f"{year:04d}-{m:02d}-01"
    end_str = f"{year:04d}-{m:02d}-{last_day:02d}"
    return year, m, start_str, end_str


def count_working_days(month_start_str: str, month_end_str: str, joined_on_str: str) -> int:
    m_start = dt_date.fromisoformat(month_start_str)
    m_end = dt_date.fromisoformat(month_end_str)
    j_date = dt_date.fromisoformat(joined_on_str)
    start = max(m_start, j_date)
    if start > m_end:
        return 0
    count = 0
    curr = start
    while curr <= m_end:
        if curr.weekday() < 5:
            count += 1
        curr += timedelta(days=1)
    return count


def serialize_employee(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": dt_to_ms(doc["created_at"]),
    }


def serialize_attendance(doc: dict[str, Any]) -> dict[str, Any]:
    raw_history = doc.get("history") or []
    serialized_history = []
    for entry in raw_history:
        raw_changes = entry.get("changes") or {}
        ser_changes = {}
        for k, v in raw_changes.items():
            f_val = v.get("from")
            t_val = v.get("to")
            if k in ("punch_in", "punch_out"):
                f_val = dt_to_ms(f_val) if isinstance(f_val, datetime) else f_val
                t_val = dt_to_ms(t_val) if isinstance(t_val, datetime) else t_val
            ser_changes[k] = {"from": f_val, "to": t_val}
        serialized_history.append(
            {
                "at": dt_to_ms(entry["at"]) if isinstance(entry.get("at"), datetime) else entry.get("at"),
                "by": entry["by"],
                "reason": entry["reason"],
                "changes": ser_changes,
            }
        )

    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc.get("punch_in")),
        "punch_out": dt_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": int(doc.get("late_minutes") or 0),
        "overtime_minutes": int(doc.get("overtime_minutes") or 0),
        "half_day": bool(doc.get("half_day") or False),
        "history": serialized_history,
    }


EpochMillisInt = Annotated[StrictInt, Field(ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)]


class EmployeeCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(..., min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def check_joined_on(cls, v: str) -> str:
        parse_strict_date(v, "joined_on")
        return v

    @field_validator("shift_end")
    @classmethod
    def check_shifts_differ(cls, v: str, info) -> str:
        start = info.data.get("shift_start", "09:30")
        if v == start:
            raise ValueError("shift_start must differ from shift_end")
        return v


class PunchInRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str = Field(..., min_length=1)
    punched_at: Optional[EpochMillisInt] = None
    status: PresenceStatusType = "PRESENT"


class PunchOutRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str = Field(..., min_length=1)
    punched_at: Optional[EpochMillisInt] = None


class RegularizeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Optional[StatusType] = None
    punch_in: Optional[EpochMillisInt] = None
    punch_out: Optional[EpochMillisInt] = None
    reason: str = Field(..., min_length=5, max_length=200)
    regularized_by: str = Field(..., min_length=1, max_length=50)


def build_attendance_list_filter(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[StatusType] = None,
) -> dict[str, Any]:
    d_from = parse_strict_date(date_from, "date_from") if date_from is not None else None
    d_to = parse_strict_date(date_to, "date_to") if date_to is not None else None
    if d_from and d_to and d_from > d_to:
        raise ValueError("date_from cannot be after date_to")

    query: dict[str, Any] = {}
    if emp_code is not None:
        query["emp_code"] = emp_code
    if status is not None:
        query["status"] = status
    if date_from is not None or date_to is not None:
        date_cond: dict[str, Any] = {}
        if date_from is not None:
            date_cond["$gte"] = date_from
        if date_to is not None:
            date_cond["$lte"] = date_to
        query["date"] = date_cond
    return query


def build_employee_monthly_pipeline(emp_code: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": start_date, "$lte": end_date}}},
        {
            "$addFields": {
                "iso_dow": {
                    "$isoDayOfWeek": {
                        "$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}
                    }
                },
                "is_present_status": {"$in": ["$status", list(PRESENCE_STATUSES)]},
                "is_half_day": {"$ifNull": ["$half_day", False]},
                "late_val": {"$ifNull": ["$late_minutes", 0]},
                "ot_val": {"$ifNull": ["$overtime_minutes", 0]},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {"$and": ["$is_present_status", {"$lte": ["$iso_dow", 5]}]},
                            {"$cond": ["$is_half_day", 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                "late_count": {"$sum": {"$cond": [{"$gt": ["$late_val", 0]}, 1, 0]}},
                "total_late_minutes": {"$sum": "$late_val"},
                "total_overtime_minutes": {"$sum": "$ot_val"},
            }
        },
    ]


def build_department_summary_pipeline(
    start_date: str, end_date: str, department: Optional[str] = None
) -> list[dict[str, Any]]:
    emp_match: dict[str, Any] = {"joined_on": {"$lte": end_date}}
    if department is not None:
        emp_match["department"] = department

    return [
        {"$match": emp_match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"emp": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$emp"]},
                                    {"$gte": ["$date", start_date]},
                                    {"$lte": ["$date", end_date]},
                                ]
                            }
                        }
                    }
                ],
                "as": "logs",
            }
        },
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {
            "$addFields": {
                "has_log": {"$ne": [{"$type": "$logs.date"}, "missing"]},
                "iso_dow": {
                    "$cond": [
                        {"$ne": [{"$type": "$logs.date"}, "missing"]},
                        {
                            "$isoDayOfWeek": {
                                "$dateFromString": {"dateString": "$logs.date", "format": "%Y-%m-%d"}
                            }
                        },
                        None,
                    ]
                },
                "is_present": {"$in": ["$logs.status", list(PRESENCE_STATUSES)]},
                "is_half": {"$ifNull": ["$logs.half_day", False]},
                "late_val": {"$ifNull": ["$logs.late_minutes", 0]},
            }
        },
        {
            "$group": {
                "_id": "$department",
                "employees": {"$addToSet": "$emp_code"},
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {"$and": ["$has_log", "$is_present", {"$lte": ["$iso_dow", 5]}]},
                            {"$cond": ["$is_half", 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "work_hours_sum": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    "$has_log",
                                    "$is_present",
                                    {"$ne": [{"$ifNull": ["$logs.work_hours", None]}, None]},
                                ]
                            },
                            "$logs.work_hours",
                            0.0,
                        ]
                    }
                },
                "work_hours_count": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    "$has_log",
                                    "$is_present",
                                    {"$ne": [{"$ifNull": ["$logs.work_hours", None]}, None]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "late_count": {"$sum": {"$cond": [{"$and": ["$has_log", {"$gt": ["$late_val", 0]}]}, 1, 0]}},
                "total_late_minutes": {"$sum": {"$cond": ["$has_log", "$late_val", 0]}},
                "leave_count": {
                    "$sum": {"$cond": [{"$and": ["$has_log", {"$eq": ["$logs.status", "LEAVE"]}]}, 1, 0]}
                },
                "on_duty_count": {
                    "$sum": {"$cond": [{"$and": ["$has_log", {"$eq": ["$logs.status", "ON_DUTY"]}]}, 1, 0]}
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": {"$size": "$employees"},
                "present_days": 1,
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": ["$work_hours_count", 0]},
                        {"$divide": ["$work_hours_sum", "$work_hours_count"]},
                        None,
                    ]
                },
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
        {"$match": {"headcount": {"$gt": 0}}},
        {"$sort": {"department": 1}},
    ]


def build_late_leaderboard_pipeline(
    start_date: str, end_date: str, limit: int, department: Optional[str] = None
) -> list[dict[str, Any]]:
    emp_lookup_pipeline: list[dict[str, Any]] = [
        {"$match": {"$expr": {"$eq": ["$emp_code", "$$emp"]}}}
    ]
    if department is not None:
        emp_lookup_pipeline.append({"$match": {"department": department}})

    return [
        {
            "$match": {
                "date": {"$gte": start_date, "$lte": end_date},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {
            "$lookup": {
                "from": "employees",
                "let": {"emp": "$_id"},
                "pipeline": emp_lookup_pipeline,
                "as": "emp_doc",
            }
        },
        {"$unwind": "$emp_doc"},
        {
            "$project": {
                "_id": 0,
                "emp_code": "$_id",
                "name": "$emp_doc.name",
                "department": "$emp_doc.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": -1},
                "output": {"rank": {"$rank": {}}},
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "emp_code": 1}},
    ]


def build_department_trend_pipeline(
    department: str, from_str: str, to_str: str
) -> list[dict[str, Any]]:
    start_dt = datetime.fromisoformat(f"{from_str}T00:00:00+00:00")
    end_dt_exclusive = datetime.fromisoformat(f"{to_str}T00:00:00+00:00") + timedelta(days=1)

    return [
        {"$match": {"date": {"$gte": "0000-01-01", "$lte": to_str}}},
        {"$limit": 1},
        {"$project": {"_id": 0, "ts": {"$literal": start_dt}}},
        {
            "$densify": {
                "field": "ts",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [start_dt, end_dt_exclusive],
                },
            }
        },
        {
            "$addFields": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$ts"}},
                "iso_dow": {"$isoDayOfWeek": "$ts"},
            }
        },
        {
            "$addFields": {
                "is_working_day": {"$lte": ["$iso_dow", 5]},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "let": {"d": "$date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", department]},
                                    {"$lte": ["$joined_on", "$$d"]},
                                ]
                            }
                        }
                    },
                    {"$project": {"_id": 0, "emp_code": 1}},
                ],
                "as": "dept_emps",
            }
        },
        {
            "$addFields": {
                "headcount": {"$size": "$dept_emps"},
                "dept_emp_codes": "$dept_emps.emp_code",
            }
        },
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"d": "$date", "codes": "$dept_emp_codes"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$date", "$$d"]},
                                    {"$in": ["$emp_code", "$$codes"]},
                                ]
                            }
                        }
                    }
                ],
                "as": "day_logs",
            }
        },
        {
            "$addFields": {
                "present_count": {
                    "$sum": {
                        "$map": {
                            "input": "$day_logs",
                            "as": "log",
                            "in": {
                                "$cond": [
                                    {"$in": ["$$log.status", list(PRESENCE_STATUSES)]},
                                    {"$cond": [{"$ifNull": ["$$log.half_day", False]}, 0.5, 1.0]},
                                    0.0,
                                ]
                            },
                        }
                    }
                },
                "late_count": {
                    "$sum": {
                        "$map": {
                            "input": "$day_logs",
                            "as": "log",
                            "in": {
                                "$cond": [
                                    {"$gt": [{"$ifNull": ["$$log.late_minutes", 0]}, 0]},
                                    1,
                                    0,
                                ]
                            },
                        }
                    }
                },
            }
        },
        {
            "$addFields": {
                "raw_rate": {
                    "$cond": [
                        {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
                        {"$divide": ["$present_count", "$headcount"]},
                        None,
                    ]
                }
            }
        },
        {
            "$addFields": {
                "attendance_rate": {
                    "$cond": [
                        {"$ne": ["$raw_rate", None]},
                        {
                            "$divide": [
                                {
                                    "$floor": {
                                        "$add": [{"$multiply": ["$raw_rate", 10000]}, 0.5]
                                    }
                                },
                                10000.0,
                            ]
                        },
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {
                    "window_avg": {
                        "$avg": "$attendance_rate",
                        "window": {"documents": [-6, 0]},
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": "$window_avg",
            }
        },
        {"$sort": {"date": 1}},
    ]


@app.get("/health", tags=["system"])
def health():
    try:
        db.command("ping")
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.post("/employees", status_code=201, tags=["employees"])
def create_employee(payload: EmployeeCreate):
    created_at = now_utc_truncated()
    doc = {
        "emp_code": payload.emp_code,
        "name": payload.name,
        "email": payload.email,
        "department": payload.department,
        "shift_start": payload.shift_start,
        "shift_end": payload.shift_end,
        "joined_on": payload.joined_on,
        "created_at": created_at,
    }
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail=f"Employee {payload.emp_code} already exists")
    return serialize_employee(doc)


@app.get("/employees", tags=["employees"])
def list_employees(
    department: Optional[str] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    query: dict[str, Any] = {}
    if department is not None:
        query["department"] = department

    total = db.employees.count_documents(query)
    skip = (page - 1) * page_size
    cursor = db.employees.find(query).sort("emp_code", ASCENDING).skip(skip).limit(page_size)
    items = [serialize_employee(d) for d in cursor]
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.post("/attendance/punch-in", status_code=201, tags=["attendance"])
def punch_in(payload: PunchInRequest):
    emp = db.employees.find_one({"emp_code": payload.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    if payload.punched_at is not None:
        punch_in_dt = ms_to_utc_dt(payload.punched_at, truncate_seconds=True)
    else:
        punch_in_dt = now_utc_truncated()

    att_date = compute_attendance_date(punch_in_dt, emp["shift_start"], emp["shift_end"])
    late_mins = compute_late_minutes(punch_in_dt, att_date, emp["shift_start"], emp["shift_end"])

    doc = {
        "emp_code": payload.emp_code,
        "date": att_date,
        "status": payload.status,
        "punch_in": punch_in_dt,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_mins,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(
            status_code=409,
            detail=f"Attendance record already exists for {payload.emp_code} on {att_date}",
        )
    return serialize_attendance(doc)


@app.post("/attendance/punch-out", status_code=200, tags=["attendance"])
def punch_out(payload: PunchOutRequest):
    emp = db.employees.find_one({"emp_code": payload.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    if payload.punched_at is not None:
        punch_out_dt = ms_to_utc_dt(payload.punched_at, truncate_seconds=True)
    else:
        punch_out_dt = now_utc_truncated()

    rec = db.attendance_logs.find_one(
        {
            "emp_code": payload.emp_code,
            "punch_in": {"$ne": None, "$lte": punch_out_dt},
        },
        sort=[("punch_in", DESCENDING)],
    )

    if not rec:
        future_rec = db.attendance_logs.find_one(
            {"emp_code": payload.emp_code, "punch_in": {"$ne": None}},
            sort=[("punch_in", DESCENDING)],
        )
        if future_rec and future_rec.get("punch_out") is None:
            raise HTTPException(status_code=422, detail=[{"msg": "punched_at must be after punch_in"}])
        raise HTTPException(status_code=404, detail="No matching punch-in record found")

    if rec.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="Already punched out for this record")

    punch_in_dt = rec["punch_in"]
    if punch_in_dt.tzinfo is None:
        punch_in_dt = punch_in_dt.replace(tzinfo=UTC)

    diff_sec = (punch_out_dt - punch_in_dt).total_seconds()
    if diff_sec <= 0:
        raise HTTPException(status_code=422, detail=[{"msg": "punched_at must be strictly after punch_in"}])
    if diff_sec > 86400:
        raise HTTPException(
            status_code=422,
            detail=[{"msg": "punched_at cannot be more than 24 hours after punch_in"}],
        )

    work_hours, ot_mins, half_day = compute_punch_out_metrics(
        punch_in_dt, punch_out_dt, rec["date"], emp["shift_start"], emp["shift_end"]
    )

    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": punch_out_dt,
                "work_hours": work_hours,
                "overtime_minutes": ot_mins,
                "half_day": half_day,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise HTTPException(status_code=409, detail="Already punched out for this record")

    return serialize_attendance(updated)


@app.get("/attendance", tags=["attendance"])
def list_attendance(
    emp_code: Optional[str] = Query(default=None),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    status: Optional[StatusType] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    try:
        query = build_attendance_list_filter(emp_code, date_from, date_to, status)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=[{"msg": str(exc)}])

    total = db.attendance_logs.count_documents(query)
    skip = (page - 1) * page_size
    cursor = (
        db.attendance_logs.find(query)
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )
    items = [serialize_attendance(d) for d in cursor]
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.patch("/attendance/{emp_code}/{date}", tags=["attendance"])
def regularize_attendance(
    payload: RegularizeRequest,
    emp_code: str = Path(...),
    date: str = Path(...),
):
    try:
        parse_strict_date(date, "date")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=[{"msg": str(exc)}])

    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not rec:
        raise HTTPException(status_code=404, detail="Attendance record not found")

    old_status = rec["status"]
    old_pi = rec.get("punch_in")
    if isinstance(old_pi, datetime) and old_pi.tzinfo is None:
        old_pi = old_pi.replace(tzinfo=UTC)
    old_po = rec.get("punch_out")
    if isinstance(old_po, datetime) and old_po.tzinfo is None:
        old_po = old_po.replace(tzinfo=UTC)
    old_wh = rec.get("work_hours")
    old_late = int(rec.get("late_minutes") or 0)
    old_ot = int(rec.get("overtime_minutes") or 0)
    old_hd = bool(rec.get("half_day") or False)

    new_status = payload.status if payload.status is not None else old_status

    if new_status in ("ABSENT", "LEAVE"):
        if payload.punch_in is not None or payload.punch_out is not None:
            raise HTTPException(
                status_code=422,
                detail=[{"msg": "Cannot supply punch_in or punch_out when status is ABSENT or LEAVE"}],
            )
        new_pi = None
        new_po = None
        new_wh = None
        new_late = 0
        new_ot = 0
        new_hd = False
    else:
        new_pi = (
            ms_to_utc_dt(payload.punch_in, truncate_seconds=True)
            if payload.punch_in is not None
            else old_pi
        )
        new_po = (
            ms_to_utc_dt(payload.punch_out, truncate_seconds=True)
            if payload.punch_out is not None
            else old_po
        )

        if new_pi is None:
            raise HTTPException(
                status_code=422,
                detail=[{"msg": "punch_in is required for presence status"}],
            )

        computed_date = compute_attendance_date(new_pi, emp["shift_start"], emp["shift_end"])
        if computed_date != date:
            raise HTTPException(
                status_code=422,
                detail=[{"msg": f"punch_in falls on attendance date {computed_date}, expected {date}"}],
            )

        new_late = compute_late_minutes(new_pi, date, emp["shift_start"], emp["shift_end"])

        if new_po is not None:
            diff_sec = (new_po - new_pi).total_seconds()
            if diff_sec <= 0 or diff_sec > 86400:
                raise HTTPException(
                    status_code=422,
                    detail=[{"msg": "punch_out must be strictly after punch_in and within 24 hours"}],
                )
            new_wh, new_ot, new_hd = compute_punch_out_metrics(
                new_pi, new_po, date, emp["shift_start"], emp["shift_end"]
            )
        else:
            new_wh = None
            new_ot = 0
            new_hd = False

    old_fields = {
        "status": old_status,
        "punch_in": old_pi,
        "punch_out": old_po,
        "work_hours": old_wh,
        "late_minutes": old_late,
        "overtime_minutes": old_ot,
        "half_day": old_hd,
    }
    new_fields = {
        "status": new_status,
        "punch_in": new_pi,
        "punch_out": new_po,
        "work_hours": new_wh,
        "late_minutes": new_late,
        "overtime_minutes": new_ot,
        "half_day": new_hd,
    }

    changes: dict[str, dict[str, Any]] = {}
    for k in ("status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day"):
        if old_fields[k] != new_fields[k]:
            changes[k] = {"from": old_fields[k], "to": new_fields[k]}

    if not changes:
        raise HTTPException(status_code=422, detail=[{"msg": "Regularization request changes nothing"}])

    history_entry = {
        "at": now_utc_truncated(),
        "by": payload.regularized_by,
        "reason": payload.reason,
        "changes": changes,
    }

    old_history_len = len(rec.get("history") or [])
    concurrency_filter: dict[str, Any] = {"_id": rec["_id"]}
    if "history" not in rec:
        concurrency_filter["history"] = {"$exists": False}
    else:
        concurrency_filter["$expr"] = {"$eq": [{"$size": {"$ifNull": ["$history", []]}}, old_history_len]}

    updated = db.attendance_logs.find_one_and_update(
        concurrency_filter,
        {
            "$set": new_fields,
            "$push": {"history": history_entry},
        },
        return_document=ReturnDocument.AFTER,
    )
    if not updated:
        raise HTTPException(status_code=409, detail="Concurrent modification detected; please retry")

    return serialize_attendance(updated)


@app.get("/analytics/employees/{emp_code}/monthly", tags=["analytics"])
def employee_monthly(
    emp_code: str = Path(...),
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    _, _, start_date, end_date = validate_month_str(month)
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    working_days = count_working_days(start_date, end_date, emp["joined_on"])
    pipeline = build_employee_monthly_pipeline(emp_code, start_date, end_date)
    results = list(db.attendance_logs.aggregate(pipeline))

    if results:
        row = results[0]
        present_days = float(row["present_days"])
        leave_days = int(row["leave_days"])
        late_count = int(row["late_count"])
        total_late_minutes = int(row["total_late_minutes"])
        total_overtime_minutes = int(row["total_overtime_minutes"])
    else:
        present_days = 0.0
        leave_days = 0
        late_count = 0
        total_late_minutes = 0
        total_overtime_minutes = 0

    present_days = round_half_up(present_days, 2)
    if working_days > 0:
        attendance_pct = round_half_up((Decimal(str(present_days)) / Decimal(working_days)) * Decimal(100), 2)
    else:
        attendance_pct = None

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late_minutes,
        "total_overtime_minutes": total_overtime_minutes,
        "attendance_pct": attendance_pct,
    }


@app.get("/analytics/departments/summary", tags=["analytics"])
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(default=None),
):
    _, _, start_date, end_date = validate_month_str(month)
    pipeline = build_department_summary_pipeline(start_date, end_date, department)
    raw_items = list(db.employees.aggregate(pipeline))

    items = []
    for r in raw_items:
        avg_wh = r.get("avg_work_hours")
        items.append(
            {
                "department": r["department"],
                "headcount": int(r["headcount"]),
                "present_days": round_half_up(r["present_days"], 2),
                "avg_work_hours": round_half_up(avg_wh, 2) if avg_wh is not None else None,
                "late_count": int(r["late_count"]),
                "total_late_minutes": int(r["total_late_minutes"]),
                "leave_count": int(r["leave_count"]),
                "on_duty_count": int(r["on_duty_count"]),
            }
        )

    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late", tags=["analytics"])
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(default=10, ge=1, le=50),
    department: Optional[str] = Query(default=None),
):
    _, _, start_date, end_date = validate_month_str(month)
    pipeline = build_late_leaderboard_pipeline(start_date, end_date, limit, department)
    raw_items = list(db.attendance_logs.aggregate(pipeline))

    items = [
        {
            "rank": int(r["rank"]),
            "emp_code": r["emp_code"],
            "name": r["name"],
            "department": r["department"],
            "total_late_minutes": int(r["total_late_minutes"]),
            "late_count": int(r["late_count"]),
        }
        for r in raw_items
    ]
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend", tags=["analytics"])
def department_trend(
    department: str = Path(...),
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
):
    try:
        d_from = parse_strict_date(from_date, "from")
        d_to = parse_strict_date(to_date, "to")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=[{"msg": str(exc)}])

    if d_to < d_from:
        raise HTTPException(status_code=422, detail=[{"msg": "'to' must be on or after 'from'"}])
    if (d_to - d_from).days + 1 > 92:
        raise HTTPException(status_code=422, detail=[{"msg": "Date range cannot exceed 92 days"}])

    dept_exists = db.employees.find_one({"department": department})
    if not dept_exists:
        raise HTTPException(status_code=404, detail="Department not found")

    pipeline = build_department_trend_pipeline(department, from_date, to_date)
    raw_items = list(db.attendance_logs.aggregate(pipeline))

    items = []
    for r in raw_items:
        att_rate = r.get("attendance_rate")
        m_avg = r.get("moving_avg_7d")
        items.append(
            {
                "date": r["date"],
                "is_working_day": bool(r["is_working_day"]),
                "headcount": int(r["headcount"]),
                "present_count": round_half_up(r["present_count"], 2),
                "late_count": int(r["late_count"]),
                "attendance_rate": round_half_up(att_rate, 4) if att_rate is not None else None,
                "moving_avg_7d": round_half_up(m_avg, 4) if m_avg is not None else None,
            }
        )

    return {"department": department, "items": items}


@app.get("/admin/explain/{endpoint}", tags=["admin"])
def explain_endpoint(
    endpoint: ExplainEndpointType = Path(...),
    emp_code: Optional[str] = Query(default=None),
    month: Optional[str] = Query(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    date_from: Optional[str] = Query(default=None),
    date_to: Optional[str] = Query(default=None),
    status: Optional[StatusType] = Query(default=None),
    from_date: Optional[str] = Query(default=None, alias="from"),
    to_date: Optional[str] = Query(default=None, alias="to"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        try:
            q_filter = build_attendance_list_filter(emp_code, date_from, date_to, status)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=[{"msg": str(exc)}])
        skip = (page - 1) * page_size
        cmd = {
            "explain": {
                "find": "attendance_logs",
                "filter": q_filter,
                "sort": {"date": -1, "emp_code": 1},
                "skip": skip,
                "limit": page_size,
            },
            "verbosity": "executionStats",
        }
        raw_explain = db.command(cmd)
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": raw_explain}

    elif endpoint == "employee_monthly":
        if not emp_code or not month:
            raise HTTPException(
                status_code=422, detail=[{"msg": "emp_code and month are required for employee_monthly"}]
            )
        _, _, start_date, end_date = validate_month_str(month)
        pipeline = build_employee_monthly_pipeline(emp_code, start_date, end_date)
        raw_explain = db.command(
            "aggregate", "attendance_logs", pipeline=pipeline, explain=True
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": raw_explain}

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(
                status_code=422, detail=[{"msg": "month is required for department_summary"}]
            )
        _, _, start_date, end_date = validate_month_str(month)
        pipeline = build_department_summary_pipeline(start_date, end_date, department)
        raw_explain = db.command(
            "aggregate", "employees", pipeline=pipeline, explain=True
        )
        return {"endpoint": endpoint, "collection": "employees", "explain": raw_explain}

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(
                status_code=422, detail=[{"msg": "month is required for late_leaderboard"}]
            )
        _, _, start_date, end_date = validate_month_str(month)
        pipeline = build_late_leaderboard_pipeline(start_date, end_date, limit, department)
        raw_explain = db.command(
            "aggregate", "attendance_logs", pipeline=pipeline, explain=True
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": raw_explain}

    elif endpoint == "department_trend":
        if not department or not from_date or not to_date:
            raise HTTPException(
                status_code=422,
                detail=[{"msg": "department, from, and to are required for department_trend"}],
            )
        try:
            d_from = parse_strict_date(from_date, "from")
            d_to = parse_strict_date(to_date, "to")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=[{"msg": str(exc)}])
        if d_to < d_from or (d_to - d_from).days + 1 > 92:
            raise HTTPException(status_code=422, detail=[{"msg": "Invalid date range"}])
        pipeline = build_department_trend_pipeline(department, from_date, to_date)
        raw_explain = db.command(
            "aggregate", "attendance_logs", pipeline=pipeline, explain=True
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": raw_explain}
