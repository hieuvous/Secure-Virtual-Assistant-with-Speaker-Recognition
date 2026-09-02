from src.assistant.router import detect_intent


CASES = [
    (
        "Bây giờ là mấy giờ?",
        "GET_TIME",
    ),
    (
        "Mấy giờ rồi?",
        "GET_TIME",
    ),
    (
        "Hôm nay thứ mấy ngày mấy?",
        "GET_DATE",
    ),
    (
        "Machine Learning học ở phòng nào?",
        "GET_COURSE_ROOM",
    ),
    (
        "Tôi còn những task nào chưa làm?",
        "GET_TASKS",
    ),
    (
        "Hôm nay tôi còn môn gì?",
        "GET_SCHEDULE",
    ),
    (
        "Lịch học ngày mai của tôi thế nào?",
        "GET_SCHEDULE",
    ),
    (
        "Xem lại các ghi chú của tôi.",
        "READ_PRIVATE_NOTE",
    ),
    (
        "Thêm task học toán vào ngày mai",
        "ADD_TASK",
    ),
    (
        "Thêm deadline báo cáo NLP vào 9h ngày 3 tháng 9",
        "ADD_TASK",
    ),
    (
        "Xóa báo cáo NLP",
        "DELETE_TASK",
    ),
    (
        "Bỏ task Báo cáo NLP đi",
        "DELETE_TASK",
    ),
]


passed = 0


for text, expected in CASES:

    result = detect_intent(
        text
    )

    actual = result[
        "intent"
    ]

    ok = (
        actual
        == expected
    )

    print(
        "PASS"
        if ok
        else "FAIL"
    )

    print(
        "Text:",
        text,
    )

    print(
        "Expected:",
        expected,
    )

    print(
        "Actual:",
        actual,
    )

    print(
        "Entities:",
        result["entities"],
    )

    print("-" * 60)

    if ok:
        passed += 1


print(
    f"{passed}/{len(CASES)} passed"
)