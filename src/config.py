from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


class ConfigurationError(RuntimeError):
    """Raised when the database configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class DatabaseConfig:
    backend: str
    supabase_url: str | None = None
    supabase_secret_key: str | None = None


def bootstrap_environment() -> Path:
    """Load the project-root .env once configuration is needed.

    Explicit process environment variables take precedence. The project .env
    supplies only missing values, which keeps deployment and test overrides
    deterministic while still making local Streamlit startup self-contained.
    """
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ConfigurationError(
            "Database configuration requires python-dotenv; install requirements.txt."
        ) from exc

    load_dotenv(dotenv_path=ENV_PATH, override=False)
    return ENV_PATH


def get_database_config() -> DatabaseConfig:
    """Bootstrap and validate the single effective database configuration."""
    bootstrap_environment()
    raw_backend = os.getenv("DATABASE_BACKEND")
    if raw_backend is None or not raw_backend.strip():
        raise ConfigurationError(
            "DATABASE_BACKEND is required. Set it explicitly to 'supabase' "
            "in the project .env or process environment."
        )

    backend = raw_backend.strip().lower()
    if backend == "sqlite":
        raise ConfigurationError(
            "SQLite runtime backend has been removed. "
            "Set DATABASE_BACKEND=supabase."
        )
    if backend != "supabase":
        raise ConfigurationError("DATABASE_BACKEND must be 'supabase'.")

    url = os.getenv("SUPABASE_URL")
    secret_key = os.getenv("SUPABASE_SECRET_KEY")
    missing = [
        name for name, value in {
            "SUPABASE_URL": url,
            "SUPABASE_SECRET_KEY": secret_key,
        }.items() if not value
    ]
    if missing:
        raise ConfigurationError(
            "Supabase is selected but missing environment variable(s): "
            + ", ".join(missing)
            + ". Add them to the project .env or process environment."
        )
    return DatabaseConfig("supabase", url, secret_key)


def supabase_project_ref(url: str | None) -> str | None:
    """Return the public project reference from a standard Supabase URL."""
    if not url:
        return None
    hostname = urlparse(url).hostname or ""
    suffix = ".supabase.co"
    return hostname[: -len(suffix)] if hostname.endswith(suffix) else None


def load_settings() -> dict:
    import yaml

    with open(ROOT / "configs" / "settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_thresholds() -> dict:
    with open(ROOT / "configs" / "thresholds.json", "r", encoding="utf-8") as f:
        return json.load(f)


def project_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p
