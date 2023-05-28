from datetime import timedelta

def calculate_total_work_hours(work_schedule):
    total_work_hours = timedelta()
    excluded_breaks = [
        (timedelta(hours=12), timedelta(hours=13, minutes=30)),
        (timedelta(hours=17, minutes=30), timedelta(hours=18))
    ]

    for start_time, end_time in work_schedule:
        start_time = timedelta(hours=start_time[0], minutes=start_time[1])
        end_time = timedelta(hours=end_time[0], minutes=end_time[1])
        work_hours = end_time - start_time
        
        for break_start, break_end in excluded_breaks:
            if start_time <= break_start < end_time or start_time <= break_end < end_time:
                break_hours = min(end_time, break_end) - max(start_time, break_start)
                work_hours -= break_hours

        total_work_hours += work_hours

    return total_work_hours.total_seconds() / 3600

# 测试例子
# work_schedule = [
#     [(8, 31), (20, 33)],
#     [(8, 40), (20, 42)],
#     [(8, 41), (17, 52)],
#     [(8, 48), (20, 39)],
#     [(8, 51), (20, 56)]
# ]
work_schedule = [
    [(9, 0), (20, 30)],  # 9
    [(9, 0), (20, 30)],  # 9
    [(9, 0), (17, 30)],  # 7
    [(9, 0), (20, 30)],  # 9
    [(9, 0), (17, 30)]   # 7
]

total_work_hours = calculate_total_work_hours(work_schedule)
print("Total work hours for the week: ", total_work_hours)
