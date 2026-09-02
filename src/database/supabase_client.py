from __future__ import annotations

from functools import lru_cache

from src.config import get_database_config


@lru_cache(maxsize=1)
def _create_supabase_client(url: str, secret_key: str):
    """Create a cached client from an already-validated configuration."""
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError(
            "Supabase support requires the 'supabase' dependency from requirements.txt."
        ) from exc

    return create_client(url, secret_key)


def get_supabase_client():
    """Return a client only when the central config selects Supabase."""
    config = get_database_config()
    if config.backend != "supabase":
        raise RuntimeError("Supabase client requested while DATABASE_BACKEND is not 'supabase'.")
    # get_database_config validates both fields for the Supabase backend.
    return _create_supabase_client(config.supabase_url, config.supabase_secret_key)
