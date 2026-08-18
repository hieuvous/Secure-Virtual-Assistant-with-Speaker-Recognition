from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_thresholds
from src.database.db import init_db
from src.database.repositories import (
    create_user,
    list_users,
    upsert_profile,
)
from src.pipeline import process_request
from src.speaker.model import get_ecapa
from src.speaker.profile import create_speaker_profile
from src.speech.preprocessing import save_uploaded_audio

init_db()
st.set_page_config(page_title="Secure Student Assistant", layout="wide")

st.title("Secure Student Virtual Assistant")
thresholds = load_thresholds()
if thresholds.get("status") == "PROVISIONAL_ONLY":
    st.warning(
        "SV/SID thresholds hiện là provisional. Phải thay bằng threshold chọn từ development data trước khi report."
    )

users = list_users()
user_options = {f"{u['id']} - {u['name']}": u["id"] for u in users}

tab_assistant, tab_enroll = st.tabs(["Assistant", "Speaker Enrollment"])

with tab_assistant:
    st.subheader("Voice Assistant")

    active_user_id = None
    if user_options:
        selected = st.selectbox(
            "Active/claimed user (chỉ dùng cho sensitive/SV)",
            ["None"] + list(user_options.keys()),
        )
        if selected != "None":
            active_user_id = user_options[selected]

    audio = st.audio_input("Nói một lệnh tiếng Việt", sample_rate=16000)

    if audio is not None and st.button("Process request", type="primary"):
        runtime = ROOT / "data" / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        path = runtime / f"query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
        save_uploaded_audio(audio, path)

        with st.spinner("Đang xử lý ASR + speaker recognition..."):
            try:
                result = process_request(str(path), active_user_id=active_user_id)
                st.write("**Transcript:**", result["transcription"]["text"])
                st.write("**Intent:**", result["nlu"]["intent"])
                st.write("**Auth:**", result["auth_requirement"])
                st.write("**Speaker result:**", result["speaker"])
                st.write("**Allowed:**", result["allowed"])
                st.success(result["action"]["message"]) if result["action"]["success"] else st.error(result["action"]["message"])
            except Exception as exc:
                st.exception(exc)

with tab_enroll:
    st.subheader("1. Create user")
    c1, c2 = st.columns(2)
    with c1:
        new_name = st.text_input("Name")
    with c2:
        new_code = st.text_input("Student code")

    if st.button("Create user"):
        if not new_name.strip():
            st.error("Name is required.")
        else:
            try:
                uid = create_user(new_name.strip(), new_code.strip() or None)
                st.success(f"Created user_id={uid}. Reload page to enroll.")
            except Exception as exc:
                st.exception(exc)

    st.divider()
    st.subheader("2. Record 5 enrollment utterances")

    users = list_users()
    user_options = {f"{u['id']} - {u['name']}": u["id"] for u in users}
    if not user_options:
        st.info("Create a user first.")
    else:
        selected_user = st.selectbox("User to enroll", list(user_options.keys()))
        uid = user_options[selected_user]

        samples = []
        for i in range(1, 6):
            sample = st.audio_input(
                f"Enrollment recording {i}/5",
                sample_rate=16000,
                key=f"enroll_{uid}_{i}",
            )
            samples.append(sample)

        if st.button("Create / replace speaker profile", type="primary"):
            if any(x is None for x in samples):
                st.error("Record all 5 samples first.")
            else:
                enroll_dir = ROOT / "data" / "users" / str(uid) / "enrollment"
                enroll_dir.mkdir(parents=True, exist_ok=True)
                paths = []
                for i, sample in enumerate(samples, start=1):
                    path = enroll_dir / f"sample_{i}.wav"
                    save_uploaded_audio(sample, path)
                    paths.append(str(path))

                with st.spinner("Extracting ECAPA embeddings..."):
                    profile = create_speaker_profile(uid, paths)
                    ecapa = get_ecapa()
                    model_version = "finetuned" if ecapa.using_finetuned else "pretrained_voxceleb"
                    upsert_profile(
                        uid,
                        profile["embedding_path"],
                        profile["num_samples"],
                        model_version,
                    )
                st.success(
                    f"Enrollment complete: {profile['num_samples']} samples, "
                    f"embedding dim={profile['embedding_dim']}, model={model_version}"
                )
