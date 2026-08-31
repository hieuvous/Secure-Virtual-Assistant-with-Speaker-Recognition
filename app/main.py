from __future__ import annotations

import json
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
    list_audit_logs,
    list_users,
    upsert_profile,

    add_task,
    get_tasks,

    add_schedule,
    get_schedule,

    add_private_note,
    get_private_notes,
)
from src.pipeline import process_request
from src.speaker.model import get_ecapa
from src.speaker.profile import create_speaker_profile
from src.speech.preprocessing import save_uploaded_audio

init_db()
st.set_page_config(page_title="Secure Student Assistant", layout="wide")

st.title("Secure Student Virtual Assistant")
thresholds = load_thresholds()

if thresholds.get("sid_status") == "NEEDS_CALIBRATION":
    st.warning(
        "SV threshold đã được calibrate từ DEV all-impostor. "
        "SID threshold vẫn cần calibrate riêng; trước đó SID chỉ dùng SV threshold làm fallback tạm."
    )

users = list_users()
user_options = {f"{u['id']} - {u['name']}": u["id"] for u in users}

tab_assistant, tab_enroll, tab_data, tab_status = st.tabs(
    [
        "Assistant",
        "Speaker Enrollment",
        "My Data",
        "Model / Evaluation",
    ]
)

with tab_assistant:
    st.subheader("Voice Assistant")
    st.caption(
        'Demo destructive command nên dùng tiêu đề trong ngoặc kép, ví dụ: '
        'Xóa deadline "Báo cáo NLP".'
    )

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
                if result["action"]["success"]:
                    st.success(result["action"]["message"])
                else:
                    st.error(result["action"]["message"])
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

    sentence_cfg_path = ROOT / "configs" / "enrollment_sentences.json"
    sentence_cfg = None
    if sentence_cfg_path.exists():
        sentence_cfg = json.loads(sentence_cfg_path.read_text(encoding="utf-8"))
        st.info(
            f"Enrollment sentence method: {sentence_cfg.get('method', 'unknown')}"
        )
    else:
        st.info(
            "Chưa có optimized enrollment sentence set. "
            "Core enrollment vẫn chạy; nếu làm Đợt 5 phoneme experiment, "
            "hãy tạo configs/enrollment_sentences.json bằng script selection."
        )

    users = list_users()
    user_options = {f"{u['id']} - {u['name']}": u["id"] for u in users}
    if not user_options:
        st.info("Create a user first.")
    else:
        selected_user = st.selectbox("User to enroll", list(user_options.keys()))
        uid = user_options[selected_user]

        samples = []
        sentences = (sentence_cfg or {}).get("sentences", [])
        for i in range(1, 6):
            label = (
                f'Câu {i}: {sentences[i-1]}'
                if len(sentences) >= i
                else f"Enrollment recording {i}/5"
            )
            sample = st.audio_input(
                label,
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

                with st.spinner("Extracting VAD + ECAPA embeddings..."):
                    profile = create_speaker_profile(uid, paths)
                    ecapa = get_ecapa()
                    model_version = (
                        "finetuned_epoch10"
                        if ecapa.using_finetuned
                        else "pretrained_voxceleb"
                    )
                    enrollment_method = (
                        (sentence_cfg or {}).get("method")
                        or "fixed_5_mean"
                    )
                    upsert_profile(
                        uid,
                        profile["embedding_path"],
                        profile["num_samples"],
                        model_version,
                        enrollment_method=enrollment_method,
                    )
                st.success(
                    f"Enrollment complete: {profile['num_samples']} samples, "
                    f"embedding dim={profile['embedding_dim']}, "
                    f"model={model_version}, method={enrollment_method}"
                )
    with tab_data:
        st.subheader("My Data")
        st.caption(
            "Nhập dữ liệu cá nhân để assistant có thể trả lời task, lịch học và ghi chú."
        )

        data_users = list_users()

        if not data_users:
            st.info("Hãy tạo user trong tab Speaker Enrollment trước.")

        else:
            data_user_options = {
                f"{u['id']} - {u['name']}": u["id"]
                for u in data_users
            }

            selected_data_user = st.selectbox(
                "User",
                list(data_user_options.keys()),
                key="my_data_user",
            )

            data_uid = data_user_options[selected_data_user]

            # =========================================================
            # TASK
            # =========================================================

            st.markdown("### Deadline / Task")

            with st.form("add_task_form"):
                task_title = st.text_input(
                    "Tên deadline",
                    placeholder="Báo cáo NLP",
                )

                task_due = st.text_input(
                    "Hạn nộp (không bắt buộc)",
                    placeholder="2026-09-05 23:59",
                )

                submit_task = st.form_submit_button("Add task")

                if submit_task:
                    if not task_title.strip():
                        st.error("Tên deadline không được để trống.")
                    else:
                        add_task(
                            data_uid,
                            task_title.strip(),
                            task_due.strip() or None,
                        )
                        st.success("Đã thêm deadline.")

            # =========================================================
            # SCHEDULE
            # =========================================================

            st.markdown("### Schedule")

            with st.form("add_schedule_form"):
                subject = st.text_input(
                    "Môn học",
                    placeholder="Machine Learning",
                )

                start_time = st.text_input(
                    "Bắt đầu",
                    placeholder="2026-09-01 09:00",
                )

                end_time = st.text_input(
                    "Kết thúc (không bắt buộc)",
                    placeholder="2026-09-01 11:00",
                )

                location = st.text_input(
                    "Phòng",
                    placeholder="I.23",
                )

                submit_schedule = st.form_submit_button("Add schedule")

                if submit_schedule:
                    if not subject.strip() or not start_time.strip():
                        st.error("Môn học và thời gian bắt đầu là bắt buộc.")
                    else:
                        add_schedule(
                            data_uid,
                            subject.strip(),
                            start_time.strip(),
                            end_time.strip() or None,
                            location.strip() or None,
                        )
                        st.success("Đã thêm lịch học.")

            # =========================================================
            # PRIVATE NOTE
            # =========================================================

            st.markdown("### Private Note")

            with st.form("add_note_form"):
                note_title = st.text_input(
                    "Tiêu đề ghi chú",
                    placeholder="Ghi chú cá nhân",
                )

                note_content = st.text_area(
                    "Nội dung",
                    placeholder="Nội dung ghi chú...",
                )

                submit_note = st.form_submit_button("Add private note")

                if submit_note:
                    if not note_content.strip():
                        st.error("Nội dung ghi chú không được để trống.")
                    else:
                        add_private_note(
                            data_uid,
                            note_title.strip() or "Ghi chú",
                            note_content.strip(),
                        )
                        st.success("Đã thêm ghi chú.")

            # =========================================================
            # VIEW CURRENT DATA
            # =========================================================

            st.divider()
            st.markdown("### Current data")

            st.write("**Tasks**")
            st.dataframe(
                get_tasks(data_uid),
                use_container_width=True,
            )

            st.write("**Schedules**")
            st.dataframe(
                get_schedule(data_uid),
                use_container_width=True,
            )

            st.write("**Private notes**")
            st.dataframe(
                get_private_notes(data_uid),
                use_container_width=True,
            )

with tab_status:
    st.subheader("Released Speaker Verification results")

    metrics_path = ROOT / "results" / "all_impostor_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        pre = metrics["pretrained"]
        ft = metrics["fine_tuned_epoch_10"]
        protocol = metrics["protocol"]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pretrained TEST EER", f"{pre['test_eer']*100:.2f}%")
            st.metric("Fine-tuned TEST EER", f"{ft['test_eer']*100:.2f}%")
        with c2:
            st.metric("Fine-tuned DEV EER", f"{ft['dev_eer']*100:.2f}%")
            st.metric("SV threshold", f"{ft['dev_threshold']:.4f}")

        st.write({
            "test_speakers": protocol["test_speakers"],
            "genuine_trials": protocol["test_genuine_trials"],
            "impostor_trials": protocol["test_impostor_trials"],
            "speaker_overlap": protocol["speaker_overlap"],
            "vad": protocol["vad"],
        })

    st.subheader("Threshold status")
    st.json(thresholds)

    st.subheader("Recent audit logs")
    st.dataframe(list_audit_logs(20), use_container_width=True)
