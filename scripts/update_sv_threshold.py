"""Apply a DEV SV calibration result to configs/thresholds.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATING_POINT_KEYS = {
    "far10": "far_le_10pct",
    "far5": "far_le_5pct",
    "far1": "far_le_1pct",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sv-json", required=True)
    parser.add_argument("--mode", choices=("eer", "far5", "far10", "far1"), default="eer")
    args = parser.parse_args()
    result_path = Path(args.sv_json)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if "eer" not in result or "eer_threshold" not in result:
        raise ValueError(f"{result_path} is not an SV DEV calibration result.")
    if "DEV" not in str(result.get("protocol", "")).upper():
        raise ValueError("Refusing to update SV threshold from a result not marked as DEV-only.")

    if args.mode == "eer":
        selected_threshold = result["eer_threshold"]
    else:
        try:
            selected_threshold = result["target_far_operating_points"][OPERATING_POINT_KEYS[args.mode]]["threshold"]
        except KeyError as exc:
            raise ValueError(f"{result_path} lacks the operating point for --mode {args.mode}.") from exc

    config_path = ROOT / "configs" / "thresholds.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["sv_threshold"] = float(selected_threshold)
    config["sv_eer"] = float(result["eer"])
    config["sv_eer_threshold"] = float(result["eer_threshold"])
    config["sv_status"] = "TUNED_FROM_DEV_PROFILE_ALL_IMPOSTOR"
    config["sv_source"] = str(result_path)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
