# Final checklist

## Requirement 1 — Speaker model
- [ ] ECAPA Epoch-10 checkpoint loads.
- [ ] 600-speaker fine-tuning configuration documented.
- [ ] Pretrained vs fine-tuned results included.
- [ ] DEV threshold selection explained.
- [ ] TEST speakers are speaker-disjoint.
- [ ] EER/FAR/FRR included.
- [ ] Final metrics use the all-impostor protocol.

## Requirement 2 — Application
- [ ] Voice input works.
- [ ] Faster-Whisper returns Vietnamese transcript.
- [ ] Enrollment works with 5 recordings.
- [ ] Speaker profile is stored.
- [ ] General function works without auth.
- [ ] Personalized function uses SID.
- [ ] Sensitive function uses SV.
- [ ] Wrong speaker is rejected.
- [ ] Unknown speaker is handled.
- [ ] SQLite stores user data.
- [ ] Audit log is recorded.

## Required tests
- [ ] `python scripts/final_system_check.py`
- [ ] `pytest -q`
- [ ] `python scripts/validate_release_artifacts.py`
- [ ] ECAPA/VAD smoke test.
- [ ] correct-speaker SV.
- [ ] wrong-speaker SV.
- [ ] known-speaker SID.
- [ ] unknown-speaker SID.
- [ ] Streamlit end-to-end demo.

## Optional / Đợt 5
- [ ] candidate sentence corpus prepared.
- [ ] phoneme-selected five sentences generated.
- [ ] random five-sentence control generated.
- [ ] same enrollment budget used.
- [ ] random vs phoneme experiment evaluated.
- [ ] optional noisy audio test.

## Submission
- [ ] README checked.
- [ ] architecture diagram/document checked.
- [ ] experiment numbers match JSON/CSV.
- [ ] report has no obsolete exploratory EER values.
- [ ] model/link included.
- [ ] dataset/link included.
- [ ] final ZIP filename uses student IDs.
- [ ] backup copy created.
