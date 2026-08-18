from __future__ import annotations

from src.assistant.router import detect_intent
from src.assistant.permissions import get_auth_requirement
from src.assistant.actions import execute_action
from src.database.repositories import (
    list_profiles,
    get_profile,
    add_audit_log,
)
from src.speaker.identification import identify_speaker
from src.speaker.verification import verify_speaker
from src.speech.asr import get_asr


def process_request(audio_path: str, active_user_id: int | None = None) -> dict:
    transcription = get_asr().transcribe(audio_path)
    nlu = detect_intent(transcription["text"])
    intent = nlu["intent"]
    auth = get_auth_requirement(intent)

    speaker_result = None
    user_id = None
    allowed = False

    if auth == "GENERAL":
        allowed = True

    elif auth == "IDENTIFICATION":
        speaker_result = identify_speaker(audio_path, list_profiles())
        allowed = not speaker_result["is_unknown"]
        user_id = speaker_result["user_id"]

    elif auth == "VERIFICATION":
        if active_user_id is not None:
            profile = get_profile(active_user_id)
            if profile:
                speaker_result = verify_speaker(
                    audio_path,
                    claimed_user_id=active_user_id,
                    embedding_path=profile["embedding_path"],
                )
                allowed = speaker_result["accepted"]
                user_id = active_user_id

    if allowed:
        action = execute_action(user_id, intent, nlu["entities"])
        result_label = "ALLOWED"
    else:
        action = {
            "success": False,
            "message": "Yêu cầu bị từ chối vì chưa xác thực/nhận diện speaker thành công.",
        }
        result_label = "REJECTED"

    add_audit_log(
        user_id=user_id,
        intent=intent,
        auth_method=auth,
        similarity_score=(speaker_result or {}).get("score"),
        threshold=(speaker_result or {}).get("threshold"),
        result=result_label,
    )

    return {
        "transcription": transcription,
        "nlu": nlu,
        "auth_requirement": auth,
        "speaker": speaker_result,
        "allowed": allowed,
        "action": action,
    }
