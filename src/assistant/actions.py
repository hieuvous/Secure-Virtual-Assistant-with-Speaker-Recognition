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
                "message": 'Để demo ổn định, hãy đặt tiêu đề trong ngoặc kép: Xóa deadline "Báo cáo NLP".',
            }
        deleted = delete_task_by_title(user_id, title)
        if deleted:
            return {
                "success": True,
                "message": f"Đã xóa deadline: {title}.",
                "data": {"deleted": deleted},
            }
        return {
            "success": False,
            "message": f"Không tìm thấy deadline có tên chính xác: {title}.",
        }

    return {
        "success": False,
        "message": "Chưa nhận diện được yêu cầu. Hãy thử một lệnh demo đơn giản hơn.",
    }
