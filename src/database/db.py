from __future__ import annotations

from src.config import get_database_config


def database_backend() -> str:
    """Return the single validated runtime database backend."""
    return get_database_config().backend


def init_db():
    """Validate and initialize the configured Supabase runtime client.

    The PostgreSQL schema is applied through ``supabase/schema.sql`` and its
    migrations; runtime code never creates or opens a local SQLite database.
    """
    from src.database.supabase_client import get_supabase_client

    get_supabase_client()
