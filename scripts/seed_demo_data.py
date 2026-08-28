import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database.db import init_db
from src.database.repositories import (
    add_private_note,
    add_schedule,
    add_task,
    create_user,
    list_users,
    upsert_course_room,
)

init_db()

users = list_users()
if users:
    uid = users[0]["id"]
    print(f"Using existing first user_id={uid}")
else:
    uid = create_user("Demo User", "DEMO001")
    print(f"Created demo user_id={uid}")

upsert_course_room("Machine Learning", "I.23")
upsert_course_room("Natural Language Processing", "E.101")

# Avoid trying to deduplicate runtime demo data; use only on a fresh DB for clean demo.
add_task(uid, "Báo cáo NLP", "2026-09-01 23:59:00")
add_private_note(
    uid,
    "Demo note",
    "Đây là ghi chú riêng để kiểm tra Speaker Verification.",
)
add_schedule(
    uid,
    "Machine Learning",
    "2026-08-27 09:00:00",
    "2026-08-27 11:00:00",
    "I.23",
)

print("Seeded course rooms, one task, one private note and one schedule.")
