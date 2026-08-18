import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database.db import init_db
from src.database.repositories import list_users, create_user, add_task, add_private_note

init_db()

users = list_users()
if users:
    uid = users[0]["id"]
    print(f"Using existing first user_id={uid}")
else:
    uid = create_user("Demo User", "DEMO001")
    print(f"Created demo user_id={uid}")

add_task(uid, "Nộp báo cáo Machine Learning", "2026-08-25 23:59:00")
add_private_note(uid, "Demo note", "Đây là ghi chú riêng dùng để kiểm tra Speaker Verification.")
print("Seeded one task and one private note.")
