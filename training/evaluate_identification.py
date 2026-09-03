"""DEV-only open-set SID threshold calibration with repeated galleries."""

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
    DEFAULT_MODEL_SOURCE, DEFAULT_VAD_SOURCE, RuntimeAlignedEmbedder, cosine_score, make_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--gallery-speakers", type=int, default=5)
    parser.add_argument("--unknown-speakers", type=int, default=10)
    parser.add_argument("--enroll-utts", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--target-unknown-far", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    return parser.parse_args()


def all_threshold_metrics(known: list[dict], unknown: list[float]) -> list[dict]:
    scores = np.asarray([row["best_score"] for row in known] + unknown, dtype=np.float32)
    thresholds = np.unique(scores)
    thresholds = np.concatenate(([scores.min() - 1e-6], thresholds, [scores.max() + 1e-6]))
    metrics = []
    for threshold in thresholds:
        known_open = float(np.mean([row["correct"] and row["best_score"] >= threshold for row in known]))
        unknown_far = float(np.mean(np.asarray(unknown) >= threshold))
        unknown_rejection = 1.0 - unknown_far
        metrics.append({
            "threshold": float(threshold), "known_open_set_accuracy": known_open,
            "unknown_false_accept_rate": unknown_far,
            "unknown_rejection_rate": unknown_rejection,
            "balanced_score": (known_open + unknown_rejection) / 2.0,
        })
    return metrics


def main() -> None:
    args = parse_args()
    if min(args.gallery_speakers, args.unknown_speakers, args.enroll_utts, args.repetitions) < 1:
        raise ValueError("Gallery/unknown speakers, enroll utterances, and repetitions must be positive.")
    if not 0 <= args.target_unknown_far <= 1:
        raise ValueError("--target-unknown-far must be between 0 and 1.")
    frame = pd.read_csv(args.dev_csv)
    if not {"speaker_id", "path"}.issubset(frame.columns):
        raise ValueError("DEV CSV must contain columns 'speaker_id' and 'path'.")

    groups = {
        str(speaker_id): group["path"].astype(str).tolist()
        for speaker_id, group in frame.groupby("speaker_id")
        if len(group) >= args.enroll_utts + 1
    }
    required_speakers = args.gallery_speakers + args.unknown_speakers
    if len(groups) < required_speakers:
        raise RuntimeError(f"Need {required_speakers} eligible DEV speakers, found {len(groups)}.")

    rng = random.Random(args.seed)
    known_trials: list[dict] = []
    unknown_scores: list[float] = []
    closed_set_correct = 0
    embedder = RuntimeAlignedEmbedder(args.checkpoint, model_source=args.model_source, vad_source=DEFAULT_VAD_SOURCE)
    try:
        speaker_ids = sorted(groups)
        for _ in range(args.repetitions):
            sampled = rng.sample(speaker_ids, required_speakers)
            gallery_speakers = sampled[:args.gallery_speakers]
            unknown_speakers = sampled[args.gallery_speakers:]
            profiles = {}
            known_queries = {}
            for speaker_id in gallery_speakers:
                paths = list(groups[speaker_id])
                rng.shuffle(paths)
                profiles[speaker_id] = make_profile([embedder.embed(path) for path in paths[:args.enroll_utts]])
                known_queries[speaker_id] = paths[args.enroll_utts:]

            for speaker_id, paths in known_queries.items():
                for path in paths:
                    query = embedder.embed(path)
                    predicted, best_score = max(
                        ((candidate, cosine_score(query, profile)) for candidate, profile in profiles.items()),
                        key=lambda item: item[1],
                    )
                    correct = predicted == speaker_id
                    closed_set_correct += int(correct)
                    known_trials.append({"correct": correct, "best_score": best_score})
            for speaker_id in unknown_speakers:
                for path in groups[speaker_id]:
                    query = embedder.embed(path)
                    unknown_scores.append(max(cosine_score(query, profile) for profile in profiles.values()))
    finally:
        embedder.close()

    if not known_trials or not unknown_scores:
        raise RuntimeError("Calibration produced no known or unknown queries.")
    points = all_threshold_metrics(known_trials, unknown_scores)
    balanced_optimum = max(points, key=lambda point: (point["balanced_score"], point["known_open_set_accuracy"]))
    constrained = [point for point in points if point["unknown_false_accept_rate"] <= args.target_unknown_far]
    if constrained:
        selected = max(
            constrained,
            key=lambda point: (point["known_open_set_accuracy"], point["unknown_rejection_rate"]),
        )
        selection_method = "maximize_known_open_set_accuracy_subject_to_unknown_far"
    else:
        selected = balanced_optimum
        selection_method = "fallback_maximize_balanced_score_no_threshold_met_target_unknown_far"

    output = {
        "sid_threshold": selected["threshold"],
        "closed_set_accuracy": closed_set_correct / len(known_trials),
        "known_open_set_accuracy": selected["known_open_set_accuracy"],
        "unknown_false_accept_rate": selected["unknown_false_accept_rate"],
        "unknown_rejection_rate": selected["unknown_rejection_rate"],
        "target_unknown_far": args.target_unknown_far,
        "selection_method": selection_method,
        "balanced_optimum": balanced_optimum,
        "gallery_speakers": args.gallery_speakers, "unknown_speakers": args.unknown_speakers,
        "enroll_utts_per_speaker": args.enroll_utts, "repetitions": args.repetitions,
        "known_queries": len(known_trials), "unknown_queries": len(unknown_scores),
        "seed": args.seed, "vad": True, "vad_source": DEFAULT_VAD_SOURCE,
        "profile": "L2-normalize each embedding, mean enrollment embeddings, L2-normalize mean",
        "similarity": "cosine", "checkpoint": str(args.checkpoint),
        "protocol": "DEV only; repeated open-set gallery/unknown sampling",
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
