"""DEV-only SV threshold calibration using runtime-aligned profiles and VAD."""

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
    parser.add_argument("--enroll-utts", type=int, default=5)
    parser.add_argument("--max-query-utts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    return parser.parse_args()


def threshold_points(genuine: np.ndarray, impostor: np.ndarray) -> list[dict]:
    scores = np.concatenate([genuine, impostor])
    thresholds = np.unique(scores)
    thresholds = np.concatenate(([scores.min() - 1e-6], thresholds, [scores.max() + 1e-6]))
    return [
        {"threshold": float(t), "far": float(np.mean(impostor >= t)), "frr": float(np.mean(genuine < t))}
        for t in thresholds
    ]


def select_far_operating_point(points: list[dict], target_far: float) -> dict:
    eligible = [point for point in points if point["far"] <= target_far]
    return min(eligible, key=lambda point: (point["frr"], point["threshold"]))


def main() -> None:
    args = parse_args()
    if args.enroll_utts < 1 or args.max_query_utts < 1:
        raise ValueError("--enroll-utts and --max-query-utts must both be positive.")
    frame = pd.read_csv(args.dev_csv)
    if not {"speaker_id", "path"}.issubset(frame.columns):
        raise ValueError("DEV CSV must contain columns 'speaker_id' and 'path'.")

    rng = random.Random(args.seed)
    eligible: dict[str, list[str]] = {}
    for speaker_id, group in frame.groupby("speaker_id"):
        paths = group["path"].astype(str).tolist()
        if len(paths) >= args.enroll_utts + 1:
            rng.shuffle(paths)
            eligible[str(speaker_id)] = paths
    if len(eligible) < 2:
        raise RuntimeError("Need at least two eligible DEV speakers for all-impostor SV trials.")

    embedder = RuntimeAlignedEmbedder(args.checkpoint, model_source=args.model_source, vad_source=DEFAULT_VAD_SOURCE)
    try:
        profiles = {
            speaker_id: make_profile([embedder.embed(path) for path in paths[:args.enroll_utts]])
            for speaker_id, paths in eligible.items()
        }
        genuine_scores: list[float] = []
        impostor_scores: list[float] = []
        for speaker_id, paths in eligible.items():
            for path in paths[args.enroll_utts:args.enroll_utts + args.max_query_utts]:
                query = embedder.embed(path)
                genuine_scores.append(cosine_score(query, profiles[speaker_id]))
                impostor_scores.extend(
                    cosine_score(query, profile)
                    for other_id, profile in profiles.items() if other_id != speaker_id
                )
    finally:
        embedder.close()

    genuine, impostor = np.asarray(genuine_scores), np.asarray(impostor_scores)
    if not len(genuine) or not len(impostor):
        raise RuntimeError("Calibration produced no genuine or impostor trials.")
    points = threshold_points(genuine, impostor)
    eer_point = min(points, key=lambda point: abs(point["far"] - point["frr"]))
    output = {
        "eer": (eer_point["far"] + eer_point["frr"]) / 2.0,
        "eer_threshold": eer_point["threshold"],
        "far_at_eer_threshold": eer_point["far"],
        "frr_at_eer_threshold": eer_point["frr"],
        "target_far_operating_points": {
            "far_le_10pct": select_far_operating_point(points, 0.10),
            "far_le_5pct": select_far_operating_point(points, 0.05),
            "far_le_1pct": select_far_operating_point(points, 0.01),
        },
        "protocol": "DEV only; normalized 5-utterance profile; all-impostor query-to-profile trials",
        "checkpoint": str(args.checkpoint), "num_dev_speakers": len(profiles),
        "enroll_utts_per_speaker": args.enroll_utts, "max_query_utts_per_speaker": args.max_query_utts,
        "genuine_trials": int(len(genuine)), "impostor_trials": int(len(impostor)),
        "vad": True, "vad_source": DEFAULT_VAD_SOURCE,
        "profile": "L2-normalize each embedding, mean enrollment embeddings, L2-normalize mean",
        "similarity": "cosine", "seed": args.seed,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
