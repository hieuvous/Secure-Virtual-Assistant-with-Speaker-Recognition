from __future__ import annotations

from datetime import datetime
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo

import re
import unicodedata

from src.database.repositories import (
    add_task,
    delete_task_by_id,
    get_course_room,
    get_private_notes,
    get_schedule,
    get_tasks,
)


VN_TZ = ZoneInfo(
    "Asia/Ho_Chi_Minh"
)


# ==========================================================
# TEXT
# ==========================================================

def _normalize_text(
    text: str,
) -> str:

    text = text.lower().strip()

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
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _title_similarity(
    spoken: str,
    actual: str,
) -> float:

    a = _normalize_text(
        spoken
    )

    b = _normalize_text(
        actual
    )

    # "Báo cáo NLP"
    # và
    # "Báo cáo môn NLP"
    # vẫn phải match tốt.
    stopwords = {
        "deadline",
        "task",
        "nhiem",
        "vu",
        "mon",
    }

    ta = (
        set(a.split())
        - stopwords
    )

    tb = (
        set(b.split())
        - stopwords
    )

    seq_score = SequenceMatcher(
        None,
        a,
        b,
    ).ratio()

    if ta and tb:

        # Một title chứa toàn bộ token
        # của title kia.
        if (
            ta <= tb
            or tb <= ta
        ):
            token_score = 0.95

        else:
            token_score = (
                len(ta & tb)
                /
                len(ta | tb)
            )

    else:
        token_score = 0.0

    return max(
        seq_score,
        token_score,
    )


# ==========================================================
# DATETIME
# ==========================================================

def _parse_stored_datetime(
    value,
) -> datetime | None:

    if not value:
        return None

    text = str(
        value
    ).strip()

    # Supabase có thể trả UTC với Z.
    if text.endswith("Z"):
        text = (
            text[:-1]
            + "+00:00"
        )

    try:
        dt = datetime.fromisoformat(
            text
        )

    except ValueError:

        dt = None

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(
                    text,
                    fmt,
                )
                break
            except ValueError:
                pass

        if dt is None:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(
            tzinfo=VN_TZ
        )

    return dt.astimezone(
        VN_TZ
    )


def _same_local_date(
    value,
    target_date: str,
) -> bool:

    dt = _parse_stored_datetime(
        value
    )

    return (
        dt is not None
        and
        dt.date().isoformat()
        == target_date
    )


def _build_due_datetime(
    entities: dict,
) -> str:

    now = datetime.now(
        VN_TZ
    )

    # User không nói ngày:
    # mặc định hôm nay.
    date_text = (
        entities.get("due_date")
        or
        now.date().isoformat()
    )

    # Không nói giờ:
    # mặc định cuối ngày.
    time_text = (
        entities.get("due_time")
        or
        "23:59"
    )

    dt = datetime.fromisoformat(
        f"{date_text}T"
        f"{time_text}:00"
    ).replace(
        tzinfo=VN_TZ
    )

    return dt.isoformat()


# ==========================================================
# FIND TASK
# ==========================================================

def _nearest_due_key(
    task: dict,
) -> tuple:

    now = datetime.now(
        VN_TZ
    )

    dt = _parse_stored_datetime(
        task.get("due_date")
    )

    if dt is None:
        return (
            2,
            float("inf"),
        )

    delta = (
        dt - now
    ).total_seconds()

    # Ưu tiên deadline tương lai.
    if delta >= 0:
        return (
            0,
            delta,
        )

    # Nếu chỉ còn deadline đã qua,
    # lấy cái gần nhất.
    return (
        1,
        abs(delta),
    )


def _find_best_task(
    user_id: int,
    spoken_title: str,
    due_date: str | None = None,
    due_time: str | None = None,
):

    tasks = get_tasks(
        user_id
    )

    candidates = []

    for task in tasks:

        similarity = _title_similarity(
            spoken_title,
            task["title"],
        )

        # Không đủ giống tên.
        if similarity < 0.68:
            continue

        dt = _parse_stored_datetime(
            task.get("due_date")
        )

        # User có nói ngày.
        if due_date:

            if (
                dt is None
                or
                dt.date().isoformat()
                != due_date
            ):
                continue

        # User có nói giờ.
        if due_time:

            if dt is None:
                continue

            hh, mm = map(
                int,
                due_time.split(":"),
            )

            if (
                dt.hour != hh
                or
                dt.minute != mm
            ):
                continue

        candidates.append(
            (
                task,
                similarity,
            )
        )

    if not candidates:
        return (
            None,
            0.0,
        )

    # Tránh chọn title khác hoàn toàn
    # chỉ vì nó gần thời gian hơn.
    best_similarity = max(
        similarity
        for _, similarity
        in candidates
    )

    strong_candidates = [
        item
        for item in candidates
        if item[1]
        >= max(
            0.68,
            best_similarity - 0.08,
        )
    ]

    # Trong các title đủ giống,
    # lấy deadline gần nhất.
    strong_candidates.sort(
        key=lambda item:
        _nearest_due_key(
            item[0]
        )
    )

    return strong_candidates[0]


# ==========================================================
# USER
# ==========================================================

def _require_user(
    user_id: int | None,
) -> dict | None:

    if user_id is None:
        return {
            "success": False,
            "message":
                "Không xác định được người dùng.",
        }

    return None


# ==========================================================
# ACTIONS
# ==========================================================

def execute_action(
    user_id: int | None,
    intent: str,
    entities: dict,
) -> dict:

    # ------------------------------------------------------
    # TIME
    # ------------------------------------------------------

    if intent == "GET_TIME":

        now = datetime.now(
            VN_TZ
        )

        return {
            "success": True,
            "message":
                f"Bây giờ là "
                f"{now.strftime('%H:%M')}.",
        }

    # ------------------------------------------------------
    # DATE
    # ------------------------------------------------------

    if intent == "GET_DATE":

        now = datetime.now(
            VN_TZ
        )

        return {
            "success": True,
            "message":
                f"Hôm nay là "
                f"{now.strftime('%d/%m/%Y')}.",
        }

    # ------------------------------------------------------
    # COURSE ROOM
    # ------------------------------------------------------

    if intent == "GET_COURSE_ROOM":

        subject = (
            entities.get("subject")
            or ""
        ).strip()

        if not subject:
            return {
                "success": False,
                "message":
                    "Không xác định được tên môn học.",
            }

        room = get_course_room(
            subject
        )

        if not room:
            return {
                "success": False,
                "message":
                    f"Chưa có thông tin phòng học "
                    f"cho môn {subject}.",
            }

        return {
            "success": True,
            "message":
                f"{subject} học tại "
                f"{room['location']}.",
            "data": room,
        }

    # ------------------------------------------------------
    # GET TASKS
    # ------------------------------------------------------

    if intent == "GET_TASKS":

        err = _require_user(
            user_id
        )

        if err:
            return err

        tasks = get_tasks(
            user_id
        )

        target_date = (
            entities.get(
                "target_date"
            )
        )

        if target_date:

            tasks = [
                task
                for task in tasks
                if _same_local_date(
                    task.get(
                        "due_date"
                    ),
                    target_date,
                )
            ]

        if not tasks:
            return {
                "success": True,

                "message": (
                    f"Không có deadline "
                    f"vào ngày {target_date}."
                    if target_date
                    else
                    "Bạn chưa có deadline đang chờ."
                ),

                "data": [],
            }

        message = (
            "Các deadline: "
            + "; ".join(
                f"{task['title']} "
                f"({task['due_date'] or 'chưa có hạn'})"
                for task in tasks
            )
        )

        return {
            "success": True,
            "message": message,
            "data": tasks,
        }

    # ------------------------------------------------------
    # GET SCHEDULE
    # ------------------------------------------------------

    if intent == "GET_SCHEDULE":

        err = _require_user(
            user_id
        )

        if err:
            return err

        # Lấy dữ liệu rồi filter theo local timezone.
        # Cách này an toàn hơn với TIMESTAMPTZ của Supabase.
        rows = get_schedule(
            user_id
        )

        target_date = (
            entities.get(
                "target_date"
            )
        )

        if target_date:

            rows = [
                row
                for row in rows
                if _same_local_date(
                    row.get(
                        "start_time"
                    ),
                    target_date,
                )
            ]

        if not rows:

            return {
                "success": True,

                "message": (
                    f"Bạn không có lịch học "
                    f"vào ngày {target_date}."
                    if target_date
                    else
                    "Bạn chưa có lịch học."
                ),

                "data": [],
            }

        message = (
            "Lịch học: "
            + "; ".join(
                f"{row['subject']} - "
                f"{row['start_time']} - "
                f"{row['location'] or 'chưa có phòng'}"
                for row in rows
            )
        )

        return {
            "success": True,
            "message": message,
            "data": rows,
        }

    # ------------------------------------------------------
    # PRIVATE NOTE
    # ------------------------------------------------------

    if intent == "READ_PRIVATE_NOTE":

        err = _require_user(
            user_id
        )

        if err:
            return err

        notes = get_private_notes(
            user_id
        )

        if not notes:
            return {
                "success": True,
                "message":
                    "Không có ghi chú riêng.",
                "data": [],
            }

        message = " | ".join(
            f"{note['title'] or 'Ghi chú'}: "
            f"{note['content']}"
            for note in notes
        )

        return {
            "success": True,
            "message": message,
            "data": notes,
        }

    # ------------------------------------------------------
    # ADD TASK
    # ------------------------------------------------------

    if intent == "ADD_TASK":

        err = _require_user(
            user_id
        )

        if err:
            return err

        title = (
            entities.get("title")
            or ""
        ).strip()

        if not title:
            return {
                "success": False,
                "message":
                    "Không xác định được "
                    "tên task/deadline.",
            }

        due_datetime = (
            _build_due_datetime(
                entities
            )
        )

        task_id = add_task(
            user_id,
            title,
            due_datetime,
        )

        due_dt = (
            _parse_stored_datetime(
                due_datetime
            )
        )

        return {
            "success": True,

            "message": (
                f'Đã thêm task "{title}" '
                f'vào '
                f'{due_dt.strftime("%H:%M %d/%m/%Y")}.'
            ),

            "data": {
                "task_id": task_id,
                "title": title,
                "due_date":
                    due_datetime,
            },
        }

    # ------------------------------------------------------
    # DELETE TASK
    # ------------------------------------------------------

    if intent == "DELETE_TASK":

        err = _require_user(
            user_id
        )

        if err:
            return err

        title = (
            entities.get("title")
            or ""
        ).strip()

        if not title:
            return {
                "success": False,
                "message":
                    "Không xác định được "
                    "tên task/deadline cần xóa.",
            }

        task, similarity = (
            _find_best_task(
                user_id=user_id,
                spoken_title=title,
                due_date=entities.get(
                    "due_date"
                ),
                due_time=entities.get(
                    "due_time"
                ),
            )
        )

        if task is None:
            return {
                "success": False,
                "message":
                    f'Không tìm thấy task/deadline '
                    f'phù hợp với "{title}".',
            }

        deleted = (
            delete_task_by_id(
                user_id,
                task["id"],
            )
        )

        if not deleted:
            return {
                "success": False,
                "message":
                    "Tìm thấy task nhưng "
                    "không thể xóa khỏi database.",
            }

        due_dt = (
            _parse_stored_datetime(
                task.get("due_date")
            )
        )

        due_text = (
            due_dt.strftime(
                "%H:%M %d/%m/%Y"
            )
            if due_dt
            else
            "không có hạn"
        )

        return {
            "success": True,

            "message": (
                f'Đã xóa '
                f'"{task["title"]}" '
                f'({due_text}).'
            ),

            "data": {
                "deleted": 1,

                "spoken_title":
                    title,

                "matched_title":
                    task["title"],

                "task_id":
                    task["id"],

                "similarity":
                    round(
                        similarity,
                        3,
                    ),

                "due_date":
                    task.get(
                        "due_date"
                    ),
            },
        }

    return {
        "success": False,
        "message":
            "Chưa nhận diện được yêu cầu.",
    }