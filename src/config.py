from __future__ import annotations

import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_settings() -> dict:
    with open(ROOT / "configs" / "settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_thresholds() -> dict:
    with open(ROOT / "configs" / "thresholds.json", "r", encoding="utf-8") as f:
        return json.load(f)


def project_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p
