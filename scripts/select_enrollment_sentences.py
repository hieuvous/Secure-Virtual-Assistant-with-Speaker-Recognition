from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.speaker.enrollment_selection import (
    greedy_phoneme_select,
    load_candidates,
    random_select,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--method", choices=["phoneme", "random"], required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-column", default="text")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--write-app-config",
        action="store_true",
        help="Also save selected texts to configs/enrollment_sentences.json",
    )
    args = p.parse_args()

    candidates = load_candidates(args.input, args.text_column)

    if args.method == "phoneme":
        rows = greedy_phoneme_select(candidates, args.n)
    else:
        rows = random_select(candidates, args.n, args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "text", "new_phonemes", "total_covered"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Selected {len(rows)} sentences by method={args.method}:")
    for r in rows:
        print(f"{r['rank']}. {r['text']}")

    if args.write_app_config:
        config_path = ROOT / "configs" / "enrollment_sentences.json"
        config_path.write_text(
            json.dumps(
                {
                    "method": args.method,
                    "source": str(Path(args.input)),
                    "sentences": [r["text"] for r in rows],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("Wrote app enrollment config:", config_path)


if __name__ == "__main__":
    main()
