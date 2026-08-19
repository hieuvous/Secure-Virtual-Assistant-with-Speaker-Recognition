from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--official-train-list", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-speakers", type=int, default=150)
    p.add_argument("--dev-speakers", type=int, default=30)
    p.add_argument("--max-utts", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def scan_by_speaker(data_root: Path) -> dict[str, list[Path]]:
    groups = defaultdict(list)
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            rel = p.relative_to(data_root)
            if len(rel.parts) < 2:
                continue
            speaker_id = rel.parts[0]
            groups[speaker_id].append(p.resolve())
    return {k: sorted(v) for k, v in groups.items()}


def speaker_from_train_line(raw: str) -> str | None:
    """Extract a speaker ID from official or path-based Vietnam-Celeb list rows."""
    tokens = raw.strip().replace("\\", "/").split()
    if not tokens:
        return None

    # Official Vietnam-Celeb train-list format: ``id00000<TAB>00000.wav``.
    if len(tokens) >= 2 and any(tokens[1].lower().endswith(ext) for ext in AUDIO_EXTS):
        return tokens[0]

    # Keep support for entries such as ``id00000/00000.wav``.
    for token in reversed(tokens):
        parts = [part for part in token.split("/") if part]
        if len(parts) >= 2 and any(token.lower().endswith(ext) for ext in AUDIO_EXTS):
            return parts[0]
    return None


def speakers_from_train_list(path: Path) -> set[str]:
    return {
        speaker_id
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if (speaker_id := speaker_from_train_line(raw)) is not None
    }


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker_id", "path"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    groups = scan_by_speaker(data_root)
    if not groups:
        raise RuntimeError(f"No audio found under {data_root}")

    eligible = sorted(groups)
    if args.official_train_list:
        allowed = speakers_from_train_list(Path(args.official_train_list))
        matched = [s for s in eligible if s in allowed]
        if matched:
            eligible = matched
        else:
            print(
                "WARNING: no speaker IDs matched the train list parser; "
                "falling back to scanned folders. Check dataset layout."
            )

    needed = args.train_speakers + args.dev_speakers
    if len(eligible) < needed:
        raise RuntimeError(
            f"Need at least {needed} eligible speakers but found {len(eligible)}."
        )

    rng.shuffle(eligible)
    train_spks = eligible[: args.train_speakers]
    dev_spks = eligible[args.train_speakers : needed]

    train_rows, val_rows, dev_rows = [], [], []

    for spk in train_spks:
        utts = groups[spk][:]
        rng.shuffle(utts)
        utts = utts[: args.max_utts]
        if len(utts) < 2:
            continue
        n_val = max(1, round(len(utts) * args.val_ratio))
        val_utts = utts[:n_val]
        train_utts = utts[n_val:]
        train_rows += [{"speaker_id": spk, "path": str(p)} for p in train_utts]
        val_rows += [{"speaker_id": spk, "path": str(p)} for p in val_utts]

    for spk in dev_spks:
        utts = groups[spk][:]
        rng.shuffle(utts)
        utts = utts[: args.max_utts]
        dev_rows += [{"speaker_id": spk, "path": str(p)} for p in utts]

    write_csv(out / "train.csv", train_rows)
    write_csv(out / "val.csv", val_rows)
    write_csv(out / "dev.csv", dev_rows)

    summary = {
        "seed": args.seed,
        "requested_train_speakers": args.train_speakers,
        "requested_dev_speakers": args.dev_speakers,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "dev_rows": len(dev_rows),
        "train_speaker_ids": train_spks,
        "dev_speaker_ids": dev_spks,
        "speaker_disjoint_train_vs_dev": set(train_spks).isdisjoint(dev_spks),
    }
    (out / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_ids")}, indent=2))


if __name__ == "__main__":
    main()
