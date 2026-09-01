from __future__ import annotations

import os
from functools import lru_cache

from src.config import ROOT


@lru_cache(maxsize=1)
def get_supabase_client():
    """Create the Supabase client only when the Supabase backend is used."""
    try:
        from supabase import create_client
    except ImportError as exc:
        raise RuntimeError(
            "Supabase support requires the 'supabase' dependency from requirements.txt."
        ) from exc

    try:
        from dotenv import load_dotenv
    except ImportError:
        # Process environment variables remain fully supported without dotenv.
        pass
    else:
        load_dotenv(ROOT / ".env")
    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")
    missing = [
        name
        for name, value in {
            "SUPABASE_URL": url,
            "SUPABASE_SECRET_KEY": secret_key,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Supabase is selected but missing environment variable(s): "
            + ", ".join(missing)
            + ". Add them to .env or the process environment."
        )

    return create_client(url, secret_key)
