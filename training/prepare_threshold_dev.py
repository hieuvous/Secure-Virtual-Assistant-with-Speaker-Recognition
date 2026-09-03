"""Create reproducible, DEV-only threshold-calibration metadata without copying audio.

Prefer ``--split-summary`` from the original fine-tuning run: its
``dev_speaker_ids`` are reused exactly. If that artifact is unavailable, the
script makes a new seeded DEV selection from the Vietnam-Celeb training audio
only, excluding supplied TEST and fine-tuning speaker IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Vietnam-Celeb audio root (speaker directories).")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--official-train-list", help="vietnam-celeb-t.txt; restricts candidates to official training audio.")
    parser.add_argument("--split-summary", help="Original split_summary.json; reuses dev_speaker_ids exactly.")
    parser.add_argument("--test-csv", help="Optional TEST metadata CSV with a speaker_id column to exclude.")
    parser.add_argument("--test-speakers-file", help="Optional one-speaker-ID-per-line TEST list to exclude.")
    parser.add_argument(
        "--exclude-speakers-file",
        help="Optional one-speaker-ID-per-line list (for example fine-tuning speakers) to exclude.",
    )
    parser.add_argument("--dev-speakers", type=int, default=37)
    parser.add_argument("--min-utts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def scan_audio(data_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            relative = path.relative_to(data_root)
            if len(relative.parts) >= 2:
                groups[relative.parts[0]].append(path.resolve())
    return {speaker: sorted(paths) for speaker, paths in groups.items()}


def official_training_audio(list_path: Path, data_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    missing = 0
    for line in list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.strip().replace("\\", "/").split()
        if len(fields) < 2:
            continue
        speaker, relative_audio = fields[0], fields[1]
        candidates = (data_root / speaker / relative_audio, data_root / relative_audio)
        audio = next((candidate for candidate in candidates if candidate.is_file()), None)
        if audio is None:
            missing += 1
        else:
            groups[speaker].append(audio.resolve())
    if missing:
        print(f"WARNING: {missing} official-training rows could not be resolved under {data_root}.")
    return {speaker: sorted(set(paths)) for speaker, paths in groups.items()}


def load_test_speakers(args: argparse.Namespace) -> set[str]:
    speakers: set[str] = set()
    if args.test_csv:
        with Path(args.test_csv).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "speaker_id" not in reader.fieldnames:
                raise ValueError("--test-csv must contain a speaker_id column.")
            speakers.update(str(row["speaker_id"]) for row in reader if row.get("speaker_id"))
    if args.test_speakers_file:
        for line in Path(args.test_speakers_file).read_text(encoding="utf-8", errors="ignore").splitlines():
            value = line.strip().replace("\\", "/").split()
            if value:
                speakers.add(value[0])
    return speakers


def load_speaker_file(path: str | None) -> set[str]:
    if not path:
        return set()
    speakers = set()
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        speaker_id = line.strip()
        if speaker_id:
            speakers.add(speaker_id)
    return speakers


def main() -> None:
    args = parse_args()
    if args.min_utts < 6:
        raise ValueError("--min-utts must be at least 6 for 5 enrollment utterances plus a query.")
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Audio data root not found: {data_root}")
    groups = (
        official_training_audio(Path(args.official_train_list), data_root)
        if args.official_train_list else scan_audio(data_root)
    )
    test_speakers = load_test_speakers(args)
    excluded_speakers = load_speaker_file(args.exclude_speakers_file)
    selected_from_summary = False
    if args.split_summary:
        summary = json.loads(Path(args.split_summary).read_text(encoding="utf-8"))
        selected = [str(speaker) for speaker in summary.get("dev_speaker_ids", [])]
        if not selected:
            raise ValueError("--split-summary does not contain dev_speaker_ids.")
        selected_from_summary = True
    else:
        eligible = sorted(
            speaker for speaker, paths in groups.items()
            if len(paths) >= args.min_utts
            and speaker not in test_speakers
            and speaker not in excluded_speakers
        )
        if len(eligible) < args.dev_speakers:
            raise RuntimeError(
                f"Need {args.dev_speakers} eligible speakers after TEST/fine-tuning exclusions, "
                f"found {len(eligible)}."
            )
        rng = random.Random(args.seed)
        rng.shuffle(eligible)
        selected = eligible[:args.dev_speakers]

    missing = [speaker for speaker in selected if speaker not in groups]
    short = [speaker for speaker in selected if speaker in groups and len(groups[speaker]) < args.min_utts]
    overlap = sorted(set(selected) & test_speakers)
    excluded_overlap = sorted(set(selected) & excluded_speakers)
    if missing or short or overlap or excluded_overlap:
        raise RuntimeError(
            f"Invalid DEV selection: missing={missing[:5]}, under_min_utts={short[:5]}, "
            f"overlaps_explicit_test={overlap[:5]}, overlaps_excluded={excluded_overlap[:5]}."
        )

    rows = [
        {"speaker_id": speaker, "path": str(path)}
        for speaker in selected for path in sorted(groups[speaker])
    ]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("speaker_id", "path"))
        writer.writeheader()
        writer.writerows(rows)

    counts = [len(groups[speaker]) for speaker in selected]
    report = {
        "output_csv": str(output_path), "source": "official_train_list" if args.official_train_list else "audio_scan",
        "reused_original_dev_partition": selected_from_summary, "seed": args.seed,
        "speakers": len(selected), "utterances": len(rows), "min_utts_per_speaker": min(counts),
        "max_utts_per_speaker": max(counts), "median_utts_per_speaker": statistics.median(counts),
        "test_speakers_supplied": len(test_speakers),
        "speaker_overlap_with_supplied_test": len(overlap),
        "excluded_speakers_count": len(excluded_speakers),
        "dev_overlap_with_excluded": len(excluded_overlap),
        "all_audio_paths_exist": all(Path(row["path"]).is_file() for row in rows),
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
