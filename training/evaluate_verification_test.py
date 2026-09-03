"""Evaluate SV on TEST with frozen DEV thresholds; never tune thresholds on TEST."""

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
    parser.add_argument("--enroll-utts", type=int, default=5)
    parser.add_argument("--max-query-utts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-source", default=DEFAULT_MODEL_SOURCE)
    return parser.parse_args()


def rate_at_threshold(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> tuple[float, float]:
    return float(np.mean(impostor >= threshold)), float(np.mean(genuine < threshold))


def descriptive_test_eer(genuine: np.ndarray, impostor: np.ndarray) -> dict:
    scores = np.concatenate([genuine, impostor])
    thresholds = np.unique(scores)
    thresholds = np.concatenate(([scores.min() - 1e-6], thresholds, [scores.max() + 1e-6]))
    points = []
    for threshold in thresholds:
        far, frr = rate_at_threshold(genuine, impostor, float(threshold))
        points.append((abs(far - frr), float(threshold), far, frr))
    _, threshold, far, frr = min(points)
    return {
        "test_eer": (far + frr) / 2.0,
        "test_eer_threshold": threshold,
        "far_at_test_eer": far,
        "frr_at_test_eer": frr,
    }


def load_frozen_thresholds(path: str) -> tuple[float, float]:
    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("sv_threshold") is None:
        raise RuntimeError(f"Frozen DEV sv_threshold is missing in {config_path}.")
    if config.get("sv_eer_threshold") is None:
        raise RuntimeError(f"Frozen DEV sv_eer_threshold is missing in {config_path}.")
    return float(config["sv_threshold"]), float(config["sv_eer_threshold"])


def main() -> None:
    args = parse_args()
    if args.enroll_utts < 1 or args.max_query_utts < 1:
        raise ValueError("--enroll-utts and --max-query-utts must be positive.")
    frozen_sv, frozen_dev_eer = load_frozen_thresholds(args.thresholds_config)
    frame = pd.read_csv(args.test_csv)
    if not {"speaker_id", "path"}.issubset(frame.columns):
        raise ValueError("TEST CSV must contain columns 'speaker_id' and 'path'.")
    paths = frame["path"].astype(str).tolist()
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"TEST CSV contains missing audio paths (first 5): {missing[:5]}")

    rng = random.Random(args.seed)
    eligible: dict[str, list[str]] = {}
    for speaker_id, group in frame.groupby("speaker_id"):
        speaker_paths = group["path"].astype(str).tolist()
        if len(speaker_paths) >= args.enroll_utts + 1:
            rng.shuffle(speaker_paths)
            eligible[str(speaker_id)] = speaker_paths
    if len(eligible) < 2:
        raise RuntimeError("Need at least two eligible TEST speakers for all-impostor SV evaluation.")

    embedder = RuntimeAlignedEmbedder(args.checkpoint, model_source=args.model_source, vad_source=DEFAULT_VAD_SOURCE)
    try:
        profiles = {
            speaker_id: make_profile([embedder.embed(path) for path in speaker_paths[:args.enroll_utts]])
            for speaker_id, speaker_paths in eligible.items()
        }
        genuine_scores: list[float] = []
        impostor_scores: list[float] = []
        for speaker_id, speaker_paths in eligible.items():
            for path in speaker_paths[args.enroll_utts:args.enroll_utts + args.max_query_utts]:
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
        raise RuntimeError("TEST evaluation produced no genuine or impostor trials.")
    descriptive = descriptive_test_eer(genuine, impostor)
    far_sv, frr_sv = rate_at_threshold(genuine, impostor, frozen_sv)
    far_eer, frr_eer = rate_at_threshold(genuine, impostor, frozen_dev_eer)
    output = {
        **descriptive,
        "test_eer_threshold_descriptive_only": True,
        "frozen_sv_threshold": frozen_sv,
        "far_at_frozen_sv_threshold": far_sv,
        "frr_at_frozen_sv_threshold": frr_sv,
        "frozen_dev_eer_threshold": frozen_dev_eer,
        "far_at_frozen_dev_eer_threshold": far_eer,
        "frr_at_frozen_dev_eer_threshold": frr_eer,
        "num_test_speakers": len(eligible),
        "enroll_utts_per_speaker": args.enroll_utts,
        "max_query_utts_per_speaker": args.max_query_utts,
        "genuine_trials": int(len(genuine)),
        "impostor_trials": int(len(impostor)),
        "vad": True,
        "vad_source": DEFAULT_VAD_SOURCE,
        "profile": "L2-normalize each embedding, mean enrollment embeddings, L2-normalize mean",
        "similarity": "cosine",
        "seed": args.seed,
        "protocol": "TEST only; frozen DEV thresholds; no threshold tuning",
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
