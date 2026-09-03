"""Manually exercise speaker identification against enrolled profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.db import init_db
from src.database.repositories import list_profiles
from src.speaker.identification import identify_speaker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()
    init_db()
    profiles = list_profiles()
    if not profiles:
        raise RuntimeError("Enroll at least 2-3 users first.")
    print(json.dumps(identify_speaker(args.audio, profiles), indent=2))


if __name__ == "__main__":
    main()
