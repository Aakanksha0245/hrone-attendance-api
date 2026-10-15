# DECISIONS.md

1. **Indexes.** On `employees`, I created a unique index on `(emp_code)` to guarantee uniqueness and fast lookups, and `(department, joined_on, emp_code)` to cover department filtering, headcount range checks (`joined_on <= date`), and sorting. On `attendance_logs`, I created a unique index on `(emp_code, date)` for natural-key uniqueness and monthly employee queries; `(date: -1, emp_code: 1)` for `listAttendance` sorting and date-range aggregations; `(status, date: -1, emp_code: 1)` for status-filtered lists; and `(emp_code, punch_in: -1)` to find the latest open punch-in during punch-out. I rejected a standalone `(department)` index on `attendance_logs` because `department` lives only on `employees`.

2. **Punch-in race.** Both requests resolve the same `emp_code` and IST attendance `date` (R1) and call `insert_one` directly without a prior `find_one`. MongoDB's unique `(emp_code, date)` index ensures exactly one insert succeeds (`201 Created`), while the second raises `DuplicateKeyError`, which my handler catches and turns into `409 Conflict`.

3. **Ties.** Using `$setWindowFields` with `$rank` sorted by `total_late_minutes: -1`, tied employees receive the same competition rank, and `$match: {rank: {$lte: limit}}` runs afterward. All employees tied at the cutoff rank are returned, ordered by `emp_code` ascending.

4. **Headcount.** `department_summary` starts its aggregation on `employees` (`joined_on <= month_end`) and uses `$lookup` with `preserveNullAndEmptyArrays: true` into `attendance_logs`, ensuring zero-log employees are counted via `$addToSet: "$emp_code"`.

5. **100x scale.** I would denormalize `department` onto `attendance_logs` and maintain pre-aggregated daily department/employee rollup collections updated on writes, avoiding cross-collection `$lookup` scans over 10M+ logs.
