"""Apply a DEV open-set SID calibration result to configs/thresholds.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid-json", required=True)
    args = parser.parse_args()
    result_path = Path(args.sid_json)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if "sid_threshold" not in result:
        raise ValueError(f"{result_path} does not contain sid_threshold.")

    config_path = ROOT / "configs" / "thresholds.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["sid_threshold"] = float(result["sid_threshold"])
    config["sid_status"] = "TUNED_FROM_DEV_OPEN_SET"
    config["sid_source"] = str(result_path)
    for source_key, config_key in (
        ("gallery_speakers", "sid_gallery_speakers"),
        ("target_unknown_far", "sid_target_unknown_far"),
    ):
        if source_key in result:
            config[config_key] = result[source_key]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
