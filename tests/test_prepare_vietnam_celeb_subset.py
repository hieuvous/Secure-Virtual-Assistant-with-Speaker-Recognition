import csv
import json
from pathlib import Path

from training.prepare_vietnam_celeb_subset import (
    main,
    speaker_from_train_line,
    utterance_from_train_line,
)


def test_parses_official_vietnam_celeb_train_list_row():
    assert speaker_from_train_line("id00000\t00000.wav") == "id00000"
    assert utterance_from_train_line("id00000\t00000.wav") == ("id00000", "00000.wav")


def test_parses_path_based_train_list_row():
    assert speaker_from_train_line("id00001/00001.wav") == "id00001"


def test_selects_requested_speaker_counts_after_utterance_filtering(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    for speaker_id, utterance_count in {"id00000": 1, "id00001": 2, "id00002": 2}.items():
        speaker_dir = data_root / speaker_id
        speaker_dir.mkdir(parents=True)
        for index in range(utterance_count):
            (speaker_dir / f"{index:05d}.wav").touch()

    output_dir = tmp_path / "metadata"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_vietnam_celeb_subset.py", "--data-root", str(data_root),
            "--output-dir", str(output_dir), "--train-speakers", "2",
            "--dev-speakers", "1", "--max-utts", "10", "--seed", "42",
        ],
    )
    main()

    summary = json.loads((output_dir / "split_summary.json").read_text(encoding="utf-8"))
    assert len(summary["train_speaker_ids"]) == 2
    assert len(summary["dev_speaker_ids"]) == 1
    assert summary["speaker_disjoint_train_vs_dev"]


def test_uses_only_utterances_listed_in_official_train_file(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    for speaker_id in ("id00000", "id00001", "id00002"):
        speaker_dir = data_root / speaker_id
        speaker_dir.mkdir(parents=True)
        for index in range(3):
            (speaker_dir / f"{index:05d}.wav").touch()

    official_list = tmp_path / "vietnam-celeb-t.txt"
    official_list.write_text(
        "\n".join(
            f"{speaker_id}\t{index:05d}.wav"
            for speaker_id in ("id00000", "id00001", "id00002")
            for index in (0, 1)
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "metadata"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_vietnam_celeb_subset.py", "--data-root", str(data_root),
            "--official-train-list", str(official_list), "--output-dir", str(output_dir),
            "--train-speakers", "2", "--dev-speakers", "1", "--max-utts", "10", "--seed", "42",
        ],
    )
    main()

    used_paths = set()
    for split in ("train", "val", "dev"):
        with (output_dir / f"{split}.csv").open(encoding="utf-8") as csv_file:
            used_paths.update(row["path"] for row in csv.DictReader(csv_file))
    assert all(Path(path).name in {"00000.wav", "00001.wav"} for path in used_paths)
