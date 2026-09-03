"""Evaluate SID on TEST with a frozen DEV threshold; never tune on TEST."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.speaker_evaluation_common import (
    DEFAULT_MODEL_SOURCE,
    DEFAULT_VAD_SOURCE,
    RuntimeAlignedEmbedder,
    cosine_score,
    make_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--thresholds-config", default="configs/thresholds.json")
    parser.add_argument("--gallery-speakers", type=int, default=5)
    parser.add_argument("--unknown-speakers", type=int, default=10)
    parser.add_argument("--enroll-utts", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.gallery_speakers, args.unknown_speakers, args.enroll_utts, args.repetitions) < 1:
        raise ValueError("Gallery/unknown speakers, enrollment utterances, and repetitions must be positive.")
    config_path = Path(args.thresholds_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("sid_threshold") is None:
        raise RuntimeError(f"Frozen DEV sid_threshold is missing in {config_path}.")
    frozen_threshold = float(config["sid_threshold"])
    frame = pd.read_csv(args.test_csv)
    if not {"speaker_id", "path"}.issubset(frame.columns):
        raise ValueError("TEST CSV must contain columns 'speaker_id' and 'path'.")
    paths = frame["path"].astype(str).tolist()
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"TEST CSV contains missing audio paths (first 5): {missing[:5]}")
    groups = {
        str(speaker_id): group["path"].astype(str).tolist()
        for speaker_id, group in frame.groupby("speaker_id")
        if len(group) >= args.enroll_utts + 1
    }
    required_speakers = args.gallery_speakers + args.unknown_speakers
    if len(groups) < required_speakers:
        raise RuntimeError(f"Need {required_speakers} eligible TEST speakers, found {len(groups)}.")

    rng = random.Random(args.seed)
    known_queries = 0
    unknown_queries = 0
    closed_set_correct = 0
    known_open_set_correct = 0
    unknown_false_accepts = 0
    embedder = RuntimeAlignedEmbedder(args.checkpoint, model_source=args.model_source, vad_source=DEFAULT_VAD_SOURCE)
    try:
        speaker_ids = sorted(groups)
        for _ in range(args.repetitions):
            sampled = rng.sample(speaker_ids, required_speakers)
            gallery_ids = sampled[:args.gallery_speakers]
            unknown_ids = sampled[args.gallery_speakers:]
            profiles = {}
            gallery_queries = {}
            for speaker_id in gallery_ids:
                speaker_paths = list(groups[speaker_id])
                rng.shuffle(speaker_paths)
                profiles[speaker_id] = make_profile(
                    [embedder.embed(path) for path in speaker_paths[:args.enroll_utts]]
                )
                gallery_queries[speaker_id] = speaker_paths[args.enroll_utts:]
            for speaker_id, speaker_paths in gallery_queries.items():
                for path in speaker_paths:
                    query = embedder.embed(path)
                    predicted, best_score = max(
                        ((candidate, cosine_score(query, profile)) for candidate, profile in profiles.items()),
                        key=lambda item: item[1],
                    )
                    correct = predicted == speaker_id
                    known_queries += 1
                    closed_set_correct += int(correct)
                    known_open_set_correct += int(correct and best_score >= frozen_threshold)
            for speaker_id in unknown_ids:
                for path in groups[speaker_id]:
                    query = embedder.embed(path)
                    best_score = max(cosine_score(query, profile) for profile in profiles.values())
                    unknown_queries += 1
                    unknown_false_accepts += int(best_score >= frozen_threshold)
    finally:
        embedder.close()

    if not known_queries or not unknown_queries:
        raise RuntimeError("TEST evaluation produced no known or unknown queries.")
    unknown_far = unknown_false_accepts / unknown_queries
    output = {
        "frozen_sid_threshold": frozen_threshold,
        "closed_set_accuracy": closed_set_correct / known_queries,
        "known_open_set_accuracy": known_open_set_correct / known_queries,
        "unknown_false_accept_rate": unknown_far,
        "unknown_rejection_rate": 1.0 - unknown_far,
        "gallery_speakers": args.gallery_speakers,
        "unknown_speakers": args.unknown_speakers,
        "enroll_utts_per_speaker": args.enroll_utts,
        "repetitions": args.repetitions,
        "known_queries": known_queries,
        "unknown_queries": unknown_queries,
        "vad": True,
        "vad_source": DEFAULT_VAD_SOURCE,
        "profile": "L2-normalize each embedding, mean enrollment embeddings, L2-normalize mean",
        "similarity": "cosine",
        "seed": args.seed,
        "protocol": "TEST only; repeated open-set evaluation with frozen DEV SID threshold; no threshold tuning",
        "threshold_tuning_on_test": False,
        "checkpoint": str(args.checkpoint),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Frozen DEV threshold used; TEST results are evaluation-only.")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
