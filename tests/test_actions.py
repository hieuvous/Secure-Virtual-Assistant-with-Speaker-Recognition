from pathlib import Path
import os

from src.database.db import init_db
from src.database import db as db_module
from src.database.repositories import (
    add_private_note,
    add_schedule,
    add_task,
    create_user,
    get_tasks,
    upsert_course_room,
)
from src.assistant.actions import execute_action


def test_core_actions_with_temp_db(tmp_path, monkeypatch):
    # Redirect database config through db_path() itself for an isolated test.
    monkeypatch.setattr(db_module, "db_path", lambda: tmp_path / "test.db")
    init_db()

    uid = create_user("Test")
    upsert_course_room("Machine Learning", "I.23")
    add_task(uid, "Báo cáo NLP")
    add_private_note(uid, "Secret", "Private content")
    add_schedule(uid, "Machine Learning", "2026-08-27 09:00:00", location="I.23")

    assert execute_action(None, "GET_TIME", {})["success"]
    assert execute_action(
        None, "GET_COURSE_ROOM", {"subject": "Machine Learning"}
    )["success"]
    assert execute_action(uid, "GET_TASKS", {})["success"]
    assert execute_action(uid, "GET_SCHEDULE", {})["success"]
    assert execute_action(uid, "READ_PRIVATE_NOTE", {})["success"]

    added = execute_action(uid, "ADD_TASK", {"title": "New Task"})
    assert added["success"]
    assert any(t["title"] == "New Task" for t in get_tasks(uid))

    deleted = execute_action(uid, "DELETE_TASK", {"title": "New Task"})
    assert deleted["success"]
    assert not any(t["title"] == "New Task" for t in get_tasks(uid))
