from src.assistant import actions


def test_core_actions_with_in_memory_repository(monkeypatch):
    """Action tests must not select any real persistence backend or network."""
    tasks = [{"id": 1, "user_id": 1, "title": "Báo cáo NLP", "due_date": None, "status": "pending"}]
    rooms = {"machine learning": {"subject": "Machine Learning", "location": "I.23"}}
    notes = [{"id": 1, "user_id": 1, "title": "Secret", "content": "Private content"}]
    schedules = [{
        "id": 1, "user_id": 1, "subject": "Machine Learning",
        "start_time": "2026-08-27 09:00:00", "location": "I.23",
    }]

    def add_task(user_id, title, due_date=None):
        task = {
            "id": len(tasks) + 1, "user_id": user_id, "title": title,
            "due_date": due_date, "status": "pending",
        }
        tasks.append(task)
        return task["id"]

    def delete_task_by_id(user_id, task_id):
        before = len(tasks)
        tasks[:] = [task for task in tasks if not (task["user_id"] == user_id and task["id"] == task_id)]
        return int(len(tasks) != before)

    monkeypatch.setattr(actions, "add_task", add_task)
    monkeypatch.setattr(actions, "delete_task_by_id", delete_task_by_id)
    monkeypatch.setattr(actions, "get_tasks", lambda user_id: [t for t in tasks if t["user_id"] == user_id])
    monkeypatch.setattr(actions, "get_course_room", lambda subject: rooms.get(subject.lower()))
    monkeypatch.setattr(actions, "get_private_notes", lambda user_id: [n for n in notes if n["user_id"] == user_id])
    monkeypatch.setattr(actions, "get_schedule", lambda user_id: [s for s in schedules if s["user_id"] == user_id])

    assert actions.execute_action(None, "GET_TIME", {})["success"]
    assert actions.execute_action(None, "GET_COURSE_ROOM", {"subject": "Machine Learning"})["success"]
    assert actions.execute_action(1, "GET_TASKS", {})["success"]
    assert actions.execute_action(1, "GET_SCHEDULE", {})["success"]
    assert actions.execute_action(1, "READ_PRIVATE_NOTE", {})["success"]

    added = actions.execute_action(1, "ADD_TASK", {"title": "New Task"})
    assert added["success"]
    assert any(task["title"] == "New Task" for task in tasks)

    deleted = actions.execute_action(1, "DELETE_TASK", {"title": "New Task"})
    assert deleted["success"]
    assert not any(task["title"] == "New Task" for task in tasks)
