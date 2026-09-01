from __future__ import annotations

from datetime import datetime

from src.database.repositories import (
    add_task,
    delete_task_by_title,
    get_course_room,
    get_private_notes,
    get_schedule,
    get_tasks,
)

import re
import unicodedata
from difflib import SequenceMatcher

from src.database.repositories import (
    add_task,
    delete_task_by_title,
    get_tasks,
    # các import khác giữ nguyên
)

def _normalize_text(text: str) -> str:
    """
    Chuẩn hóa để so sánh:
    - lowercase
    - bỏ dấu tiếng Việt
    - bỏ ký tự đặc biệt
    - chuẩn hóa khoảng trắng
    """
    text = text.lower().strip()

    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _find_similar_task(user_id: int, spoken_title: str):
    """
    Tìm task gần giống nhất với title do ASR nhận được.
    """

    tasks = get_tasks(user_id)

    if not tasks:
        return None, 0.0

    target = _normalize_text(spoken_title)

    best_task = None
    best_score = 0.0

    for task in tasks:
        actual_title = task["title"]

        score = SequenceMatcher(
            None,
            target,
            _normalize_text(actual_title)
        ).ratio()

        if score > best_score:
            best_score = score
            best_task = task

    return best_task, best_score


def _require_user(user_id: int | None) -> dict | None:
    if user_id is None:
        return {"success": False, "message": "Không xác định được người dùng."}
    return None


def execute_action(user_id: int | None, intent: str, entities: dict) -> dict:
    if intent == "GET_TIME":
        now = datetime.now()
        return {
            "success": True,
            "message": f"Bây giờ là {now.strftime('%H:%M')}.",
        }

    if intent == "GET_DATE":
        now = datetime.now()
        return {
            "success": True,
            "message": f"Hôm nay là {now.strftime('%d/%m/%Y')}.",
        }

    if intent == "GET_COURSE_ROOM":
        subject = (entities.get("subject") or "").strip()
        if not subject:
            return {
                "success": False,
                "message": 'Hãy nói rõ tên môn, ví dụ: Môn "Machine Learning" học phòng nào?',
            }
        room = get_course_room(subject)
        if not room:
            return {
                "success": False,
                "message": f"Chưa có thông tin phòng học cho môn {subject}.",
            }
        return {
            "success": True,
            "message": f"{subject} học tại {room['location']}.",
            "data": room,
        }

    if intent == "GET_TASKS":
        err = _require_user(user_id)
        if err:
            return err
        tasks = get_tasks(user_id)
        if not tasks:
            return {
                "success": True,
                "message": "Bạn chưa có deadline đang chờ.",
                "data": [],
            }
        msg = "Các deadline đang chờ: " + "; ".join(
            f"{t['title']} ({t['due_date'] or 'chưa có hạn'})" for t in tasks
        )
        return {"success": True, "message": msg, "data": tasks}

    if intent == "GET_SCHEDULE":
        err = _require_user(user_id)
        if err:
            return err
        rows = get_schedule(user_id)
        if not rows:
            return {
                "success": True,
                "message": "Bạn chưa có lịch học.",
                "data": [],
            }
        msg = "Lịch học: " + "; ".join(
            f"{r['subject']} - {r['start_time']} - {r['location'] or 'chưa có phòng'}"
            for r in rows
        )
        return {"success": True, "message": msg, "data": rows}

    if intent == "READ_PRIVATE_NOTE":
        err = _require_user(user_id)
        if err:
            return err
        notes = get_private_notes(user_id)
        if not notes:
            return {
                "success": True,
                "message": "Không có ghi chú riêng.",
                "data": [],
            }
        msg = " | ".join(
            f"{n['title'] or 'Ghi chú'}: {n['content']}" for n in notes
        )
        return {"success": True, "message": msg, "data": notes}

    if intent == "ADD_TASK":
        err = _require_user(user_id)
        if err:
            return err
        title = (entities.get("title") or "").strip()
        if not title:
            return {
                "success": False,
                "message": 'Để demo ổn định, hãy đặt tiêu đề trong ngoặc kép: Thêm deadline "Báo cáo NLP".',
            }
        task_id = add_task(user_id, title)
        return {
            "success": True,
            "message": f"Đã thêm deadline: {title}.",
            "data": {"task_id": task_id, "title": title},
        }

    if intent == "DELETE_TASK":
        err = _require_user(user_id)
        if err:
            return err

        title = (entities.get("title") or "").strip()

        if not title:
            return {
                "success": False,
                "message": (
                    "Không xác định được tên deadline cần xóa. "
                    "Ví dụ: Xóa deadline Báo cáo NLP."
                ),
            }

        # ---------------------------------------------------------
        # 1. Thử xóa chính xác trước
        # ---------------------------------------------------------
        deleted = delete_task_by_title(user_id, title)

        if deleted:
            return {
                "success": True,
                "message": f"Đã xóa deadline: {title}.",
                "data": {
                    "deleted": deleted,
                    "matched_title": title,
                    "match_type": "exact",
                },
            }

        # ---------------------------------------------------------
        # 2. Nếu ASR nhận sai một chút → fuzzy matching
        # ---------------------------------------------------------
        best_task, similarity = _find_similar_task(user_id, title)

        # 0.80 đủ để xử lý NLP -> NLB nhưng tránh match quá xa
        if best_task is not None and similarity >= 0.80:
            actual_title = best_task["title"]

            deleted = delete_task_by_title(
                user_id,
                actual_title
            )

            if deleted:
                return {
                    "success": True,
                    "message": (
                        f'Đã xóa deadline: "{actual_title}". '
                        f'(Lệnh nhận được: "{title}")'
                    ),
                    "data": {
                        "deleted": deleted,
                        "spoken_title": title,
                        "matched_title": actual_title,
                        "similarity": round(similarity, 3),
                        "match_type": "fuzzy",
                    },
                }

        # ---------------------------------------------------------
        # 3. Không tìm được task đủ giống
        # ---------------------------------------------------------
        return {
            "success": False,
            "message": (
                f'Không tìm thấy deadline phù hợp với "{title}".'
            ),
            "data": {
                "best_similarity": round(similarity, 3),
            },
        }

    return {
        "success": False,
        "message": "Chưa nhận diện được yêu cầu. Hãy thử một lệnh demo đơn giản hơn.",
    }
