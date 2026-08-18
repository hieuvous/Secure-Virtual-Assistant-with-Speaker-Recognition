import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database.db import init_db, db_path

init_db()
print(f"Database initialized: {db_path()}")
