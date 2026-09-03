from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import re
import unicodedata


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


# Chỉ thêm những lỗi ASR mà nhóm đã THỰC SỰ quan sát nhiều lần.
ASR_ALIASES = {
    "dech loi": "deadline",
    "det lai": "deadline",
    "det line": "deadline",
    "det loi": "deadline",
}


def _norm(text: str) -> str:
    text = text.lower().strip()

     # NFD không tự chuyển "đ" thành "d"
    text = text.replace("đ", "d")
    text = unicodedata.normalize(
        "NFD",
        text,
    )

    text = "".join(
        c
        for c in text
        if unicodedata.category(c) != "Mn"
    )

    text = re.sub(
        r"[^\w\s:]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    for wrong, correct in ASR_ALIASES.items():
        text = text.replace(
            wrong,
            correct,
        )

    # Ví dụ ASR trả "N L P".
    text = re.sub(
        r"\bn\s+l\s+p\b",
        "nlp",
        text,
    )

    return text




def _quoted(text: str) -> str | None:
    match = re.search(
        r'["“](.+?)["”]',
        text,
    )

    return (
        match.group(1).strip()
        if match
        else None
    )


# ==========================================================
# DATE
# ==========================================================

def _extract_date(
    text: str,
) -> str | None:

    t = _norm(text)

    today = datetime.now(
        VN_TZ
    ).date()

    relative_dates = {
        "hom qua": -1,
        "hom nay": 0,
        "ngay mai": 1,
        "ngay kia": 2,
    }

    for phrase, offset in relative_dates.items():
        if phrase in t:
            return (
                today
                + timedelta(days=offset)
            ).isoformat()

    match = re.search(
        r"\bngay\s+(\d{1,2})"
        r"(?:\s+thang\s+(\d{1,2}))?"
        r"(?:\s+nam\s+(\d{4}))?\b",
        t,
    )

    if not match:
        return None

    day = int(
        match.group(1)
    )

    has_month = (
        match.group(2)
        is not None
    )

    has_year = (
        match.group(3)
        is not None
    )

    month = (
        int(match.group(2))
        if has_month
        else today.month
    )

    year = (
        int(match.group(3))
        if has_year
        else today.year
    )

    try:
        candidate = datetime(
            year,
            month,
            day,
            tzinfo=VN_TZ,
        ).date()

    except ValueError:
        return None

    # "ngày 3" nhưng ngày 3 tháng này đã qua:
    # hiểu là ngày 3 tháng sau.
    if (
        not has_month
        and candidate < today
    ):
        next_month = month + 1
        next_year = year

        if next_month == 13:
            next_month = 1
            next_year += 1

        try:
            candidate = datetime(
                next_year,
                next_month,
                day,
                tzinfo=VN_TZ,
            ).date()
        except ValueError:
            return None

    # "ngày 3 tháng 9" đã qua nhưng không nói năm:
    # hiểu là năm sau.
    elif (
        has_month
        and not has_year
        and candidate < today
    ):
        try:
            candidate = datetime(
                year + 1,
                month,
                day,
                tzinfo=VN_TZ,
            ).date()
        except ValueError:
            return None

    return candidate.isoformat()


# ==========================================================
# TIME
# ==========================================================

def _extract_time(
    text: str,
) -> str | None:

    t = _norm(text)

    # 9h
    # 9h30
    # 9 giờ
    # lúc 9 giờ 30
    match = re.search(
        r"\b(?:vao\s+|luc\s+)?"
        r"(\d{1,2})\s*(?:h|gio)"
        r"(?:\s*(\d{1,2}))?"
        r"(?:\s*phut)?\b",
        t,
    )

    # 09:30
    if not match:
        match = re.search(
            r"\b(\d{1,2}):(\d{2})\b",
            t,
        )

    if not match:
        return None

    hour = int(
        match.group(1)
    )

    minute = int(
        match.group(2) or 0
    )

    if not (
        0 <= hour <= 23
        and
        0 <= minute <= 59
    ):
        return None

    return (
        f"{hour:02d}:"
        f"{minute:02d}"
    )


# ==========================================================
# TASK TITLE
# ==========================================================

def _extract_task_title(
    text: str,
) -> str | None:

    quoted = _quoted(text)

    if quoted:
        return quoted

    title = text.strip()

    # Xóa động từ đầu command.
    title = re.sub(
        r"^\s*"
        r"(?:thêm|them|tạo|tao|"
        r"xóa|xoa|hủy|huy|bỏ|bo)"
        r"(?:\s+giúp\s+tôi|\s+giup\s+toi)?"
        r"\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # Bỏ "cái", "deadline", "task", "mới cho"...
    title = re.sub(
        r"^\s*"
        r"(?:cái|cai)?\s*"
        r"(?:deadline|task|"
        r"nhiệm\s*vụ|nhiem\s+vu)?"
        r"\s*"
        r"(?:mới\s+cho|moi\s+cho|cho)?"
        r"\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # Xóa giờ khỏi title.
    title = re.sub(
        r"\b(?:vào|vao|lúc|luc)\s+"
        r"\d{1,2}\s*"
        r"(?:h|giờ|gio)"
        r"(?:\s*\d{1,2})?"
        r"(?:\s*phút|\s*phut)?\b",
        " ",
        title,
        flags=re.IGNORECASE,
    )

    title = re.sub(
        r"\b\d{1,2}:\d{2}\b",
        " ",
        title,
    )

    # Xóa ngày cụ thể.
    title = re.sub(
        r"\b(?:ngày|ngay)\s+"
        r"\d{1,2}"
        r"(?:\s+(?:tháng|thang)\s+\d{1,2})?"
        r"(?:\s+(?:năm|nam)\s+\d{4})?",
        " ",
        title,
        flags=re.IGNORECASE,
    )

    # Xóa ngày tương đối.
    title = re.sub(
        r"\b(?:"
        r"hôm qua|hom qua|"
        r"hôm nay|hom nay|"
        r"ngày mai|ngay mai|"
        r"ngày kia|ngay kia"
        r")\b",
        " ",
        title,
        flags=re.IGNORECASE,
    )

    # "thêm task NLP vào lịch"
    title = re.sub(
        r"\b(?:vào|vao)\s+"
        r"(?:lịch|lich)\b",
        " ",
        title,
        flags=re.IGNORECASE,
    )

    title = re.sub(
        r"\b(?:vào|vao|lúc|luc)\b\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    title = re.sub(
        r"\b(?:đi|di)\b\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    title = title.strip(
        " ,.;:-"
    )

    return (
        title
        if title
        else None
    )


# ==========================================================
# COURSE SUBJECT
# ==========================================================

def _extract_course_subject(
    text: str,
) -> str | None:

    quoted = _quoted(text)

    if quoted:
        return quoted

    patterns = [
        # Môn Machine Learning học phòng nào?
        r"(?:môn|mon)\s+(.+?)\s+"
        r"(?:học|hoc)\s+"
        r"(?:(?:ở|o)\s+)?"
        r"phòng\s+nào",

        # Phòng học môn Machine Learning ở đâu?
        r"phòng\s+học\s+"
        r"(?:môn\s+|mon\s+)?"
        r"(.+?)"
        r"(?:\s+ở\s+đâu|\s+o\s+dau|\?|$)",

        # Machine Learning học ở phòng nào?
        r"(.+?)\s+"
        r"(?:học|hoc)\s+"
        r"(?:(?:ở|o)\s+)?"
        r"phòng\s+nào",

        # Môn Machine Learning ở đâu?
        r"(?:môn|mon)\s+(.+?)"
        r"(?:\s+ở\s+đâu|\s+o\s+dau|\?|$)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            subject = (
                match.group(1)
                .strip(" ?.,")
            )

            if subject:
                return subject

    return None


# ==========================================================
# INTENT
# ==========================================================

def detect_intent(
    text: str,
) -> dict:

    t = _norm(text)

    entities = {
        "raw_text": text,
    }

    parsed_date = _extract_date(
        text
    )

    parsed_time = _extract_time(
        text
    )

    # ------------------------------------------------------
    # 1. TIME
    # ------------------------------------------------------

    if (
        "may gio" in t
        or "gio roi" in t
        or "gio hien tai" in t
    ):
        intent = "GET_TIME"

    # ------------------------------------------------------
    # 2. DATE
    # ------------------------------------------------------

    elif any(
        key in t
        for key in [
            "ngay may",
            "thu may",
            "ngay bao nhieu",
            "hom nay la ngay",
            "hom nay ngay",
        ]
    ):
        intent = "GET_DATE"

    # ------------------------------------------------------
    # 3. COURSE ROOM
    # ------------------------------------------------------

    elif (
        "phong hoc" in t
        or "phong nao" in t
        or "hoc o phong" in t
        or "hoc phong" in t
    ):
        intent = "GET_COURSE_ROOM"

        subject = _extract_course_subject(
            text
        )

        if subject:
            entities["subject"] = subject

    # ------------------------------------------------------
    # 4. PRIVATE NOTE
    # ------------------------------------------------------

    elif (
        "ghi chu" in t
        and any(
            key in t
            for key in [
                "doc",
                "xem",
                "mo",
            ]
        )
    ):
        intent = "READ_PRIVATE_NOTE"

    # ------------------------------------------------------
    # 5. DELETE TASK
    # ------------------------------------------------------

    elif re.search(
        r"\b(xoa|huy|bo)\b",
        t,
    ):
        intent = "DELETE_TASK"

        title = _extract_task_title(
            text
        )

        if title:
            entities["title"] = title

        if parsed_date:
            entities["due_date"] = parsed_date

        if parsed_time:
            entities["due_time"] = parsed_time

    # ------------------------------------------------------
    # 6. ADD TASK
    # ------------------------------------------------------

    elif re.search(
        r"\b(them|tao)\b",
        t,
    ):
        intent = "ADD_TASK"

        title = _extract_task_title(
            text
        )

        if title:
            entities["title"] = title

        if parsed_date:
            entities["due_date"] = parsed_date

        if parsed_time:
            entities["due_time"] = parsed_time

    # ------------------------------------------------------
    # 7. GET TASKS
    # ------------------------------------------------------

    elif any(
        key in t
        for key in [
            "deadline",
            "task",
            "nhiem vu",
            "con viec",
        ]
    ):
        intent = "GET_TASKS"

        if parsed_date:
            entities["target_date"] = parsed_date

    # ------------------------------------------------------
    # 8. GET SCHEDULE
    # ------------------------------------------------------

    elif (
        "lich hoc" in t
        or "thoi khoa bieu" in t
        or "toi hoc mon gi" in t
        or "hoc mon gi" in t
        or "co mon nao" in t
        or "con mon gi" in t

        # Hôm nay tôi còn môn gì?
        or (
            parsed_date is not None
            and "mon" in t
            and any(
                key in t
                for key in [
                    "hoc",
                    "con",
                    "co",
                ]
            )
        )
    ):
        intent = "GET_SCHEDULE"

        if parsed_date:
            entities["target_date"] = parsed_date

    else:
        intent = "UNKNOWN"

    return {
        "intent": intent,
        "entities": entities,

        "confidence": (
            1.0
            if intent != "UNKNOWN"
            else 0.0
        ),
    }