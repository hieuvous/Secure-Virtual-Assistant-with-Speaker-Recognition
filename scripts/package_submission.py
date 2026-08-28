from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_PARTS = {
    ".venv", "venv", "__pycache__", ".pytest_cache", ".git",
    "data/runtime", "data/users",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def excluded(path: Path, include_model: bool) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(rel == x or rel.startswith(x + "/") for x in EXCLUDE_PARTS):
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    if not include_model and rel == "models/ecapa_vietnamceleb_epoch10.pt":
        return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student-ids", nargs="+", required=True)
    p.add_argument("--include-model", action="store_true")
    p.add_argument("--output-dir", default="submission")
    args = p.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "_".join(args.student_ids) + ".zip"
    out = out_dir / name

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in ROOT.rglob("*"):
            if path.is_file() and not excluded(path, args.include_model):
                z.write(path, path.relative_to(ROOT))

    print("Created:", out)
    if not args.include_model:
        print(
            "Model was excluded. If model/dataset are submitted by Drive/Hugging Face link, "
            "put the required link text file beside the submission according to lecturer rules."
        )


if __name__ == "__main__":
    main()
