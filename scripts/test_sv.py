"""Manually exercise speaker verification for one enrolled user."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.db import init_db
from src.database.repositories import get_profile
from src.speaker.verification import verify_speaker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()
    init_db()
    profile = get_profile(args.user_id)
    if not profile:
        raise RuntimeError("User has no enrolled speaker profile.")
    print(json.dumps(verify_speaker(args.audio, args.user_id, profile["embedding"]), indent=2))


if __name__ == "__main__":
    main()
