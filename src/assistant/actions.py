from __future__ import annotations

from datetime import datetime

from src.database.repositories import get_tasks, get_private_notes


def execute_action(user_id: int | None, intent: str, entities: dict) -> dict:
    if intent == "GET_TIME":
        return {"success": True, "message": datetime.now().strftime("Bây giờ là %H:%M.")}

    if intent == "GET_DATE":
        return {"success": True, "message": datetime.now().strftime("Hôm nay là %d/%m/%Y.")}

    if intent == "GET_TASKS":
        if user_id is None:
            return {"success": False, "message": "Không xác định được người dùng."}
        tasks = get_tasks(user_id)
        if not tasks:
            return {"success": True, "message": "Bạn chưa có deadline đang chờ.", "data": []}
        msg = "Các deadline đang chờ: " + "; ".join(
            f"{t['title']} ({t['due_date'] or 'chưa có hạn'})" for t in tasks
        )
        return {"success": True, "message": msg, "data": tasks}

    if intent == "READ_PRIVATE_NOTE":
        if user_id is None:
            return {"success": False, "message": "Không xác định được người dùng."}
        notes = get_private_notes(user_id)
        if not notes:
            return {"success": True, "message": "Không có ghi chú riêng.", "data": []}
        msg = " | ".join(f"{n['title']}: {n['content']}" for n in notes)
        return {"success": True, "message": msg, "data": notes}

    if intent in {"ADD_TASK", "DELETE_TASK", "UPDATE_SCHEDULE", "GET_SCHEDULE", "GET_COURSE_ROOM"}:
        return {
            "success": False,
            "message": f"{intent} đã có route/auth contract nhưng action chi tiết sẽ hoàn thiện ở đợt code tiếp theo.",
        }

    return {
        "success": False,
        "message": "Chưa nhận diện được yêu cầu. Hãy thử một lệnh demo đơn giản hơn.",
    }
