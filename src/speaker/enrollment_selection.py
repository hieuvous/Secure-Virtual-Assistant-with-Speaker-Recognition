from __future__ import annotations

import csv
import random
import re
from pathlib import Path


def load_candidates(path: str | Path, text_column: str = "text") -> list[str]:
    """
    Accept:
    - .txt: one sentence per line
    - .csv: a column named `text` by default
    """
    path = Path(path)
    if path.suffix.lower() == ".txt":
        rows = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if text_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"CSV must contain column '{text_column}'. Found: {reader.fieldnames}"
                )
            rows = [
                (row.get(text_column) or "").strip()
                for row in reader
                if (row.get(text_column) or "").strip()
            ]
    else:
        raise ValueError("Candidate file must be .txt or .csv.")

    # De-duplicate while preserving order.
    return list(dict.fromkeys(rows))


def _clean_tokens(tokens: list[str]) -> set[str]:
    out = set()
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"[\s,.;:!?\"'()\[\]{}_-]+", token):
            continue
        out.add(token)
    return out


def phoneme_set(text: str) -> set[str]:
    """
    Vietnamese G2P using Viphoneme.

    This is an experiment utility only. It is intentionally isolated from
    the core assistant so a Viphoneme installation problem cannot break SV/SID.
    """
    try:
        from viphoneme import vi2IPA_split
    except ImportError as exc:
        raise RuntimeError(
            "Phoneme experiment dependency is missing. Run: "
            "pip install -r requirements-experiments.txt"
        ) from exc

    raw = vi2IPA_split(text, "/")
    return _clean_tokens(raw.split("/"))


def random_select(
    candidates: list[str],
    n: int = 5,
    seed: int = 42,
) -> list[dict]:
    if len(candidates) < n:
        raise ValueError(f"Need at least {n} candidate sentences.")
    rng = random.Random(seed)
    selected = rng.sample(candidates, n)
    return [
        {"rank": i + 1, "text": text, "new_phonemes": None, "total_covered": None}
        for i, text in enumerate(selected)
    ]


def greedy_phoneme_select(
    candidates: list[str],
    n: int = 5,
) -> list[dict]:
    """
    Project implementation proposal:
    repeatedly select the sentence that adds the most unseen phoneme tokens.

    This is NOT claimed to be an exact reproduction of the paper's private/full
    optimization algorithm.
    """
    if len(candidates) < n:
        raise ValueError(f"Need at least {n} candidate sentences.")

    phonemes = {text: phoneme_set(text) for text in candidates}
    covered: set[str] = set()
    remaining = list(candidates)
    result = []

    for rank in range(1, n + 1):
        scored = []
        for text in remaining:
            new_set = phonemes[text] - covered
            # Tie-break: prefer more total phonemes, then shorter text.
            score = (len(new_set), len(phonemes[text]), -len(text))
            scored.append((score, text, new_set))

        scored.sort(reverse=True, key=lambda x: x[0])
        _, best_text, new_set = scored[0]

        covered |= phonemes[best_text]
        result.append({
            "rank": rank,
            "text": best_text,
            "new_phonemes": len(new_set),
            "total_covered": len(covered),
        })
        remaining.remove(best_text)

    return result
