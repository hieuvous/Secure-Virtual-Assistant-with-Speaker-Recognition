from src.config import load_thresholds
from src.speaker.embedding import embedding_to_numpy, extract_embedding
from src.speaker.scoring import cosine_score


def identify_embedding(query, profiles: list[dict], threshold: float) -> dict:
    if not profiles:
        return {
            "user_id": None, "name": None, "score": float("-inf"),
            "threshold": float(threshold), "is_unknown": True, "ranking": [],
        }

    ranking = [
        {
            "user_id": int(profile["user_id"]),
            "name": profile.get("name"),
            "score": cosine_score(query, profile["embedding"]),
        }
        for profile in profiles
    ]
    ranking.sort(key=lambda row: row["score"], reverse=True)
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
    if threshold is None:
        threshold = load_thresholds().get("sid_threshold")
        if threshold is None:
            raise RuntimeError(
                "SID threshold has not been calibrated. Run training/evaluate_identification.py first."
            )

    loaded = []
    for profile in profiles:
        embedding = profile.get("embedding")
        if embedding is not None:
            loaded.append({
                "user_id": int(profile["user_id"]),
                "name": profile.get("name"),
                "embedding": embedding_to_numpy(embedding),
            })

    result = identify_embedding(
        extract_embedding(audio_path, use_vad=True), loaded, float(threshold)
    )
    result["sid_threshold_calibrated"] = True
    result["threshold_source"] = "SID_DEV"
    return result
