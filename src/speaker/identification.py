from pathlib import Path
import numpy as np

from src.config import load_thresholds
from src.speaker.embedding import extract_embedding
from src.speaker.scoring import cosine_score


def identify_embedding(query, profiles: list[dict], threshold: float) -> dict:
    if not profiles:
        return {
            "user_id": None, "name": None, "score": float("-inf"),
            "threshold": float(threshold), "is_unknown": True, "ranking": []
        }

    ranking = []
    for p in profiles:
        ranking.append({
            "user_id": int(p["user_id"]),
            "name": p.get("name"),
            "score": cosine_score(query, p["embedding"]),
        })
    ranking.sort(key=lambda x: x["score"], reverse=True)

    best = ranking[0]
    unknown = best["score"] < threshold
    return {
        "user_id": None if unknown else best["user_id"],
        "name": None if unknown else best.get("name"),
        "score": float(best["score"]),
        "threshold": float(threshold),
        "is_unknown": bool(unknown),
        "ranking": ranking,
    }


def identify_speaker(audio_path: str, profiles: list[dict], threshold: float | None = None) -> dict:
    tcfg = load_thresholds()
    calibrated = True

    if threshold is None:
        threshold = tcfg.get("sid_threshold")
        if threshold is None:
            # App can be smoke-tested before SID calibration, but this MUST NOT
            # be reported as the final SID threshold.
            threshold = float(tcfg["sv_threshold"])
            calibrated = False

    loaded = []
    for p in profiles:
        path = Path(p["embedding_path"])
        if path.exists():
            loaded.append({
                "user_id": int(p["user_id"]),
                "name": p.get("name"),
                "embedding": np.load(path),
            })

    result = identify_embedding(
        extract_embedding(audio_path, use_vad=True),
        loaded,
        float(threshold),
    )
    result["sid_threshold_calibrated"] = calibrated
    result["threshold_source"] = "SID_DEV" if calibrated else "PROVISIONAL_SV_FALLBACK"
    return result
