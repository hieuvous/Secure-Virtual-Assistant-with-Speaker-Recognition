"""
Fast final check that does NOT require microphone or ASR inference.

It checks:
- required release artifacts;
- model/config metric consistency;
- DB schema;
- router/action logic;
- SID calibration status;
- optional enrollment experiment outputs.

Use `smoke_test_ecapa.py` separately for real VAD+ECAPA inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.router import detect_intent
from src.config import load_thresholds
from src.database.db import init_db, db_path


def check(condition: bool, ok: str, fail: str):
    if condition:
        print("[PASS]", ok)
        return True
    print("[FAIL]", fail)
    return False


def main():
    passed = True

    required = [
        ROOT / "models" / "ecapa_vietnamceleb_epoch10.pt",
        ROOT / "models" / "config.json",
        ROOT / "results" / "all_impostor_metrics.json",
        ROOT / "results" / "pretrained_vs_finetuned.csv",
        ROOT / "app" / "main.py",
        ROOT / "src" / "pipeline.py",
    ]
    for path in required:
        passed &= check(path.exists(), f"Found {path.relative_to(ROOT)}", f"Missing {path}")

    config = json.loads((ROOT / "models" / "config.json").read_text())
    metrics = json.loads((ROOT / "results" / "all_impostor_metrics.json").read_text())
    tuned = metrics["fine_tuned_epoch_10"]

    passed &= check(
        abs(config["verification_threshold"] - tuned["dev_threshold"]) < 1e-12,
        "SV threshold matches release metrics",
        "SV threshold mismatch",
    )
    passed &= check(
        metrics["protocol"]["speaker_overlap"] == 0,
        "Final TEST is speaker-disjoint",
        "Speaker overlap is not zero",
    )

    init_db()
    passed &= check(db_path().exists(), f"Database initialized at {db_path()}", "DB init failed")

    router_cases = {
        "Bây giờ là mấy giờ?": "GET_TIME",
        'Môn "Machine Learning" học phòng nào?': "GET_COURSE_ROOM",
        "Tôi còn deadline nào?": "GET_TASKS",
        'Xóa deadline "Báo cáo NLP"': "DELETE_TASK",
        "Đọc ghi chú riêng của tôi": "READ_PRIVATE_NOTE",
    }
    for text, expected in router_cases.items():
        got = detect_intent(text)["intent"]
        passed &= check(got == expected, f"Router: {expected}", f"Router expected {expected}, got {got}")

    t = load_thresholds()
    passed &= check(
        str(t.get("sv_status", "")).startswith("TUNED_FROM_DEV"),
        "SV threshold is calibrated",
        "SV threshold status is not final",
    )

    if str(t.get("sid_status", "")).startswith("TUNED_FROM_DEV") and t.get("sid_threshold") is not None:
        print("[PASS] SID threshold calibrated:", t["sid_threshold"])
    else:
        print("[WARN] SID threshold is not calibrated yet. Run training/evaluate_identification.py.")

    selected = ROOT / "configs" / "enrollment_sentences.json"
    if selected.exists():
        print("[PASS] Enrollment sentence config exists:", selected)
    else:
        print("[INFO] No optimized enrollment sentence config yet. This is optional for core requirement.")

    print()
    if passed:
        print("CORE STATIC CHECK: PASS")
        print("Next: pytest -> ECAPA/VAD smoke test -> enrollment -> SV -> SID -> Streamlit.")
    else:
        raise SystemExit("CORE STATIC CHECK: FAILED")


if __name__ == "__main__":
    main()
