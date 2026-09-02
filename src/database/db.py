from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.config import ConfigurationError, get_database_config, load_settings, project_path


def database_backend() -> str:
    """Return the already-bootstrapped, validated effective backend."""
    return get_database_config().backend


def db_path() -> Path:
    return project_path(load_settings()["database"]["path"])


@contextmanager
def connect():
    if database_backend() != "sqlite":
        raise ConfigurationError(
            "SQLite connection requested while DATABASE_BACKEND is not 'sqlite'."
        )
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
    if database_backend() == "supabase":
        # The Supabase schema is applied manually from supabase/schema.sql.
        # Initializing the client here makes missing credentials fail early.
        from src.database.supabase_client import get_supabase_client

        get_supabase_client()
        return

    schema_path = Path(__file__).with_name("schema.sql")
    with connect() as conn:
        conn.executescript(schema_path.read_text(encoding="utf-8"))
