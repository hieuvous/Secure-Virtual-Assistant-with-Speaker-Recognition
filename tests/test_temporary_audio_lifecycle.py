from pathlib import Path

import pytest
import torch

from src import pipeline
from src.speaker import enrollment
from src.speech import vad


def test_enrollment_temp_files_exist_during_profile_creation_and_are_removed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    observed_paths: list[Path] = []
    upserts = []

    def fake_create_profile(user_id, audio_paths, sentence_ids=None):
        paths = [Path(path) for path in audio_paths]
        assert all(path.exists() for path in paths)
        observed_paths.extend(paths)
        return {
            "user_id": user_id,
            "embedding": [1.0] + [0.0] * 191,
            "num_samples": len(paths),
            "embedding_dim": 192,
        }

    monkeypatch.setattr(enrollment, "create_speaker_profile", fake_create_profile)
    monkeypatch.setattr(enrollment, "upsert_profile", lambda *args, **kwargs: upserts.append((args, kwargs)))

    profile = enrollment.enroll_speaker_from_recordings(
        7, [b"one", b"two", b"three", b"four", b"five"], model_version="test-model"
    )

    assert profile["num_samples"] == 5
    assert len(upserts) == 1
    assert all(not path.exists() for path in observed_paths)
    assert not (tmp_path / "data" / "users").exists()


def test_enrollment_temp_files_are_removed_after_profile_exception(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    observed_paths: list[Path] = []

    def failing_create_profile(_user_id, audio_paths, sentence_ids=None):
        observed_paths.extend(Path(path) for path in audio_paths)
        assert all(path.exists() for path in observed_paths)
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(enrollment, "create_speaker_profile", failing_create_profile)
    monkeypatch.setattr(enrollment, "upsert_profile", lambda *_args, **_kwargs: pytest.fail("must not upsert"))

    with pytest.raises(RuntimeError, match="embedding failed"):
        enrollment.enroll_speaker_from_recordings(
            7, [b"one", b"two", b"three", b"four", b"five"], model_version="test-model"
        )

    assert all(not path.exists() for path in observed_paths)
    assert not (tmp_path / "data" / "users").exists()


def test_request_temp_wav_exists_during_processing_and_is_removed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    observed_paths: list[Path] = []

    def fake_process_request(audio_path, active_user_id=None):
        path = Path(audio_path)
        assert path.exists()
        observed_paths.append(path)
        return {"ok": True, "active_user_id": active_user_id}

    monkeypatch.setattr(pipeline, "process_request", fake_process_request)
    assert pipeline.process_request_recording(b"query", active_user_id=9) == {
        "ok": True, "active_user_id": 9,
    }
    assert all(not path.exists() for path in observed_paths)
    assert not (tmp_path / "data" / "runtime").exists()


def test_request_temp_wav_is_removed_after_processing_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    observed_paths: list[Path] = []

    def failing_process_request(audio_path, active_user_id=None):
        path = Path(audio_path)
        observed_paths.append(path)
        assert path.exists()
        raise RuntimeError("request failed")

    monkeypatch.setattr(pipeline, "process_request", failing_process_request)
    with pytest.raises(RuntimeError, match="request failed"):
        pipeline.process_request_recording(b"query")
    assert all(not path.exists() for path in observed_paths)
    assert not (tmp_path / "data" / "runtime").exists()


def _vad_service_with(model):
    service = vad.VADService.__new__(vad.VADService)
    service.model = model
    return service


def test_vad_canonical_wav_is_temporary_and_trimmed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    waveform = torch.ones((1, 16_000), dtype=torch.float32)
    observed_paths: list[Path] = []

    class Model:
        def get_speech_segments(self, canonical_path):
            path = Path(canonical_path)
            observed_paths.append(path)
            assert path.exists()
            return [[0.0, 1.0]]

    monkeypatch.setattr(vad, "load_mono_16k", lambda _path: waveform)
    monkeypatch.setattr(vad.torchaudio, "save", lambda path, *_args: Path(path).write_bytes(b"wav"))

    trimmed = _vad_service_with(Model()).trim("input.wav")
    assert torch.equal(trimmed, waveform)
    assert all(not path.exists() for path in observed_paths)
    assert not (tmp_path / "data" / "runtime" / "vad").exists()


def test_vad_failure_falls_back_and_removes_canonical_wav(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    waveform = torch.ones((1, 16_000), dtype=torch.float32)
    observed_paths: list[Path] = []

    class FailingModel:
        def get_speech_segments(self, canonical_path):
            path = Path(canonical_path)
            observed_paths.append(path)
            assert path.exists()
            raise RuntimeError("VAD unavailable")

    monkeypatch.setattr(vad, "load_mono_16k", lambda _path: waveform)
    monkeypatch.setattr(vad.torchaudio, "save", lambda path, *_args: Path(path).write_bytes(b"wav"))

    assert torch.equal(_vad_service_with(FailingModel()).trim("input.wav"), waveform)
    assert all(not path.exists() for path in observed_paths)
    assert not (tmp_path / "data" / "runtime" / "vad").exists()
