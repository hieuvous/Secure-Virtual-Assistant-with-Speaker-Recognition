"""Temporary-audio enrollment service for the persistent Supabase profile."""

from __future__ import annotations

from collections.abc import Sequence

from src.database.repositories import upsert_profile
from src.speaker.profile import create_speaker_profile
from src.speech.temporary_audio import temporary_audio_paths


def enroll_speaker_from_recordings(
    user_id: int,
    recordings: Sequence[bytes],
    *,
    model_version: str,
    enrollment_method: str = "fixed_5_mean",
    sentence_ids: list[int] | None = None,
) -> dict:
    """Create and persist a final profile without retaining raw recordings.

    The profile math remains in ``create_speaker_profile``: each ECAPA vector
    is L2-normalized, then averaged, then L2-normalized again.
    """
    with temporary_audio_paths(recordings, prefix="va_enrollment_") as paths:
        profile = create_speaker_profile(
            user_id,
            [str(path) for path in paths],
            sentence_ids=sentence_ids,
        )
        upsert_profile(
            user_id,
            profile["embedding"],
            profile["num_samples"],
            model_version,
            enrollment_method=enrollment_method,
        )
        return profile
