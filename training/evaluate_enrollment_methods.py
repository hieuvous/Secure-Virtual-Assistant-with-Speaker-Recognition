"""
Compare enrollment sentence-selection methods using real recorded enrollment audio.

Input enrollment CSV:
speaker_id,method,audio_path

Example methods:
random
phoneme

Each speaker/method should have the SAME enrollment budget (recommended 5 recordings).

Input query CSV:
speaker_id,audio_path

The script reports:
- SID closed-set accuracy
- SV EER (all gallery profiles as genuine/impostor comparisons)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.speaker.embedding import extract_embedding, l2_normalize
from src.speaker.scoring import cosine_score


def profile(embeddings):
    return l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))


def eer(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enrollment-csv", required=True)
    p.add_argument("--query-csv", required=True)
    p.add_argument("--output-csv", required=True)
    args = p.parse_args()

    enr = pd.read_csv(args.enrollment_csv)
    qry = pd.read_csv(args.query_csv)
    required_enr = {"speaker_id", "method", "audio_path"}
    required_qry = {"speaker_id", "audio_path"}

    if not required_enr.issubset(enr.columns):
        raise ValueError(f"Enrollment CSV needs columns: {sorted(required_enr)}")
    if not required_qry.issubset(qry.columns):
        raise ValueError(f"Query CSV needs columns: {sorted(required_qry)}")

    cache = {}

    def emb(path):
        path = str(path)
        if path not in cache:
            cache[path] = extract_embedding(path, use_vad=True)
        return cache[path]

    rows = []

    for method in sorted(enr["method"].astype(str).unique()):
        sub = enr[enr["method"].astype(str) == method]
        profiles = {}

        for spk, group in sub.groupby("speaker_id"):
            profiles[str(spk)] = profile(
                [emb(p) for p in group["audio_path"].astype(str)]
            )

        # Only evaluate queries whose speaker is in the gallery.
        qsub = qry[qry["speaker_id"].astype(str).isin(profiles)]
        if qsub.empty:
            raise RuntimeError(f"No valid queries for method={method}")

        correct = 0
        total = 0
        scores = []
        labels = []

        for _, r in qsub.iterrows():
            true_spk = str(r["speaker_id"])
            q = emb(r["audio_path"])
            ranked = sorted(
                ((spk, cosine_score(q, pvec)) for spk, pvec in profiles.items()),
                key=lambda x: x[1],
                reverse=True,
            )
            correct += int(ranked[0][0] == true_spk)
            total += 1

            for cand_spk, score in ranked:
                scores.append(score)
                labels.append(1 if cand_spk == true_spk else 0)

        sv_eer, threshold = eer(scores, labels)
        rows.append({
            "method": method,
            "gallery_speakers": len(profiles),
            "query_count": total,
            "sid_closed_set_accuracy": correct / total,
            "sv_eer": sv_eer,
            "sv_eer_threshold": threshold,
        })

    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
