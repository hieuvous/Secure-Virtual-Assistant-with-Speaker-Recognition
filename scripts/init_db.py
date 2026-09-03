import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database.db import init_db, database_backend

init_db()
print(f"Database client initialized: {database_backend()}")
