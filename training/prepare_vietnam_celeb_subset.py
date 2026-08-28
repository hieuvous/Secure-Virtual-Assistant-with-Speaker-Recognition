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
    p.add_argument("--official-train-list")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-speakers", type=int, default=600)
    p.add_argument("--dev-speakers", type=int, default=50)
    p.add_argument("--max-utts", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def speakers_from_train_list(path: Path) -> set[str]:
    speakers = set()
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().replace("\\", "/").split()
        if parts:
            speakers.add(parts[0].split("/")[0])
    return speakers


def official_audio_map(path: Path, data_root: Path) -> dict[str, list[Path]]:
    """Parse Vietnam-Celeb-T rows like: id00000<TAB>00000.wav."""
    groups = defaultdict(list)
    missing = 0

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().replace("\\", "/").split()
        if len(parts) < 2:
            continue

        speaker_id, rel_audio = parts[0], parts[1]
        candidates = [
            data_root / speaker_id / rel_audio,
            data_root / rel_audio,
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is not None:
            groups[speaker_id].append(found.resolve())
        else:
            missing += 1

    total = sum(len(v) for v in groups.values())
    print(f"Restricted to {total} audio files in the official train list.")
    if missing:
        print(f"WARNING: {missing} official-list rows could not be resolved.")
    return {k: sorted(set(v)) for k, v in groups.items()}


def scan_audio(data_root: Path) -> dict[str, list[Path]]:
    groups = defaultdict(list)
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            rel = p.relative_to(data_root)
            if len(rel.parts) >= 2:
                groups[rel.parts[0]].append(p.resolve())
    return {k: sorted(v) for k, v in groups.items()}


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["speaker_id", "path"])
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.official_train_list:
        groups = official_audio_map(Path(args.official_train_list), data_root)
    else:
        groups = scan_audio(data_root)

    eligible = sorted(k for k, v in groups.items() if len(v) >= 2)
    needed = args.train_speakers + args.dev_speakers
    if len(eligible) < needed:
        raise RuntimeError(f"Need {needed} eligible speakers, found {len(eligible)}.")

    rng.shuffle(eligible)
    train_spks = eligible[:args.train_speakers]
    dev_spks = eligible[args.train_speakers:needed]

    train_rows, val_rows, dev_rows = [], [], []

    for spk in train_spks:
        paths = groups[spk][:]
        rng.shuffle(paths)
        paths = paths[:args.max_utts]
        n_val = max(1, round(len(paths) * args.val_ratio))
        n_val = min(n_val, len(paths) - 1)
        val_paths, train_paths = paths[:n_val], paths[n_val:]
        train_rows += [{"speaker_id": spk, "path": str(p)} for p in train_paths]
        val_rows += [{"speaker_id": spk, "path": str(p)} for p in val_paths]

    for spk in dev_spks:
        paths = groups[spk][:]
        rng.shuffle(paths)
        paths = paths[:args.max_utts]
        dev_rows += [{"speaker_id": spk, "path": str(p)} for p in paths]

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
        "speaker_disjoint_train_vs_dev": set(train_spks).isdisjoint(dev_spks),
        "train_speaker_ids": train_spks,
        "dev_speaker_ids": dev_spks,
    }
    (out / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {k: v for k, v in summary.items() if not k.endswith("_ids")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
