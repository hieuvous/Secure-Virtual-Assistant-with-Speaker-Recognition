from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier


TARGET_SR = 16000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dev-csv", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output-json", required=True)
    p.add_argument("--max-utts-per-speaker", type=int, default=8)
    p.add_argument("--fixed-threshold", type=float, default=None,
                   help="Optional fixed threshold for additional FAR/FRR reporting.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    p.add_argument("--cache-dir", default="./pretrained_ecapa_eval",
                   help="SpeechBrain pretrained-model cache directory.")
    return p.parse_args()


def load_audio(path: str):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.float()


def normalize(x):
    x = np.asarray(x, np.float32).reshape(-1)
    return x / max(np.linalg.norm(x), 1e-12)


def embed(model, path: str, device: str):
    wav = load_audio(path).to(device)
    with torch.inference_mode():
        e = model.encode_batch(wav, normalize=False).squeeze().cpu().numpy()
    return normalize(e)


def cosine(a, b):
    return float(np.dot(normalize(a), normalize(b)))


def compute_eer(scores: np.ndarray, labels: np.ndarray):
    # Positive label = 1, accept if score >= threshold.
    thresholds = np.unique(scores)
    thresholds = np.concatenate(
        ([scores.min() - 1e-6], thresholds, [scores.max() + 1e-6])
    )
    fars, frrs = [], []
    pos = labels == 1
    neg = labels == 0

    for t in thresholds:
        far = float(np.mean(scores[neg] >= t)) if neg.any() else 0.0
        frr = float(np.mean(scores[pos] < t)) if pos.any() else 0.0
        fars.append(far)
        frrs.append(frr)

    fars = np.asarray(fars)
    frrs = np.asarray(frrs)
    idx = int(np.argmin(np.abs(fars - frrs)))
    eer = float((fars[idx] + frrs[idx]) / 2.0)
    return {
        "threshold": float(thresholds[idx]),
        "eer": eer,
        "far": float(fars[idx]),
        "frr": float(frrs[idx]),
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    model = EncoderClassifier.from_hparams(
        source=args.model_source,
        savedir=str(Path(args.cache_dir)),
        run_opts={"device": device},
    )

    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = payload.get("embedding_model", payload)
        model.mods.embedding_model.load_state_dict(state, strict=False)
        print("Loaded fine-tuned checkpoint:", args.checkpoint)
    else:
        print("Evaluating pretrained baseline.")

    df = pd.read_csv(args.dev_csv)
    df["speaker_id"] = df["speaker_id"].astype(str)

    by_spk = {}
    for spk, group in df.groupby("speaker_id"):
        paths = list(group["path"].astype(str))
        rng.shuffle(paths)
        by_spk[spk] = paths[: args.max_utts_per_speaker]

    unique_paths = sorted({p for paths in by_spk.values() for p in paths})
    embeddings = {}
    for i, path in enumerate(unique_paths, start=1):
        embeddings[path] = embed(model, path, device)
        if i % 25 == 0 or i == len(unique_paths):
            print(f"Embedded {i}/{len(unique_paths)}")

    positive_pairs = []
    for spk, paths in by_spk.items():
        positive_pairs.extend((a, b, 1) for a, b in itertools.combinations(paths, 2))

    speakers = [s for s, paths in by_spk.items() if paths]
    negative_pairs = []
    target_neg = len(positive_pairs)
    seen = set()

    while len(negative_pairs) < target_neg:
        s1, s2 = rng.sample(speakers, 2)
        a = rng.choice(by_spk[s1])
        b = rng.choice(by_spk[s2])
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        negative_pairs.append((a, b, 0))

    trials = positive_pairs + negative_pairs
    rng.shuffle(trials)

    scores = np.asarray([cosine(embeddings[a], embeddings[b]) for a, b, _ in trials])
    labels = np.asarray([y for _, _, y in trials], dtype=np.int64)

    metrics = compute_eer(scores, labels)
    if args.fixed_threshold is not None:
        pos = labels == 1
        neg = labels == 0
        metrics.update(
            {
                "fixed_threshold": float(args.fixed_threshold),
                "far_at_fixed_threshold": (
                    float(np.mean(scores[neg] >= args.fixed_threshold)) if neg.any() else 0.0
                ),
                "frr_at_fixed_threshold": (
                    float(np.mean(scores[pos] < args.fixed_threshold)) if pos.any() else 0.0
                ),
            }
        )
    metrics.update(
        {
            "positive_trials": int((labels == 1).sum()),
            "negative_trials": int((labels == 0).sum()),
            "num_dev_speakers": len(by_spk),
            "checkpoint": args.checkpoint or "pretrained",
            "seed": args.seed,
        }
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
