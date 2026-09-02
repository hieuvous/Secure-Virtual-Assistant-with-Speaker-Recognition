"""Print the effective database configuration without connecting or exposing secrets."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ConfigurationError, ENV_PATH, get_database_config, supabase_project_ref
from src.database.db import db_path


def main() -> int:
    print(f"Project root: {ROOT}")
    print(f".env found: {ENV_PATH.is_file()}")
    try:
        config = get_database_config()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}")
        return 2

    print(f"Configured backend: {config.backend}")
    print(f"Effective backend: {config.backend}")
    if config.backend == "supabase":
        print(f"Supabase URL configured: {bool(config.supabase_url)}")
        print(f"Supabase secret configured: {bool(config.supabase_secret_key)}")
        print(f"Supabase project ref: {supabase_project_ref(config.supabase_url) or 'unavailable'}")
        print("SQLite path: N/A")
    else:
        print("Supabase URL configured: False")
        print("Supabase secret configured: False")
        print("Supabase project ref: N/A")
        print(f"SQLite path: {db_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
