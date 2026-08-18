from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.config import load_settings, project_path


def db_path() -> Path:
    return project_path(load_settings()["database"]["path"])


@contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    schema_path = Path(__file__).with_name("schema.sql")
    with connect() as conn:
        conn.executescript(schema_path.read_text(encoding="utf-8"))
