# Final Run Guide — easiest order

## Stage 0 — Open in VS Code

Open the whole folder:

```text
secure-student-assistant-final-dot5-dot6/
```

Use Python 3.10.

## Stage 1 — Create environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Meaning:
- `venv`: creates an isolated Python environment;
- `Activate.ps1`: switches terminal to that environment;
- `pip install`: installs SpeechBrain, Streamlit, Faster-Whisper, etc.

## Stage 2 — Static project check

```powershell
python scripts/final_system_check.py
```

Meaning:
- checks final model/config/result files exist;
- checks released SV threshold matches metrics;
- initializes SQLite;
- checks important intent routing;
- warns if SID threshold is still pending.

This does NOT use the microphone.

## Stage 3 — Unit tests

```powershell
pytest -q tests
```

Meaning:
- tests cosine/SV/SID logic;
- tests intent routing;
- tests DB actions;
- tests deterministic random enrollment selection.

All tests should pass before real microphone testing.

## Stage 4 — Check final checkpoint structure

```powershell
python scripts/validate_release_artifacts.py
```

Expected key values:
- epoch 10;
- embedding dimension 192;
- 600 training speakers;
- SV threshold 0.1566438227891922.

## Stage 5 — First real audio/model test

Use a clean WAV of yourself:

```powershell
python scripts/smoke_test_ecapa.py test.wav
```

Meaning:
- loads VAD;
- loads base SpeechBrain ECAPA architecture;
- overlays `ecapa_vietnamceleb_epoch10.pt`;
- trims speech;
- extracts 192-D normalized embedding.

First run may download and cache:
- `speechbrain/spkrec-ecapa-voxceleb`
- `speechbrain/vad-crdnn-libriparty`

After cache is present, the demo can run offline.

Expected:
```text
ECAPA + VAD OK
Using fine-tuned: True
Embedding shape: (192,)
L2 norm: ~1.0
```

## Stage 6 — Fresh demo DB

For the cleanest demo, delete `data/app.db` if it contains old test data.

Then:

```powershell
python scripts/init_db.py
python scripts/seed_demo_data.py
```

Meaning:
- creates SQLite tables;
- creates/uses one demo user;
- adds sample course room;
- adds task;
- adds private note;
- adds schedule.

## Stage 7 — Run Streamlit

```powershell
streamlit run app/main.py
```

Open the local address displayed by Streamlit.

## Stage 8 — Enrollment test

Create at least **2 users**.

Each user records **5 separate utterances**.

Expected:
- speaker profile `.npy` created;
- model = `finetuned_epoch10`;
- embedding dimension = 192.

## Stage 9 — Speaker Verification tests

Correct person claiming user 1:

```powershell
python scripts/test_sv.py --user-id 1 --audio user1_query.wav
```

Expected normally:
```text
accepted: true
```

Wrong person claiming user 1:

```powershell
python scripts/test_sv.py --user-id 1 --audio other_person.wav
```

Expected normally:
```text
accepted: false
```

Do not manually change the SV threshold to make a single demo recording pass.
Inspect the score and keep the DEV-calibrated threshold.

## Stage 10 — SID tests

```powershell
python scripts/test_sid.py --audio user1_query.wav
python scripts/test_sid.py --audio unknown_person.wav
```

Known user should identify correctly.

Unknown speaker should be rejected **only after SID threshold calibration is complete**.

If output says:
```text
sid_threshold_calibrated=false
```
run the SID calibration from Đợt 3 before final demo.

## Stage 11 — End-to-end voice demo

In Streamlit, test:

### General
- `Bây giờ là mấy giờ?`
- `Môn "Machine Learning" học phòng nào?`

No authentication expected.

### Personalized / SID
- `Tôi còn deadline nào?`
- `Hôm nay tôi học môn gì?`

SID should identify the speaker and return that user's data.

### Sensitive / SV
Choose the claimed user first.

- `Đọc ghi chú riêng của tôi`
- `Thêm deadline "Báo cáo NLP"`
- `Xóa deadline "Báo cáo NLP"`

The action is executed only when SV passes.

## Stage 12 — Optional phoneme experiment

Core project does NOT need this package.

```powershell
pip install -r requirements-experiments.txt
```

Prepare a candidate transcript pool, then:

```powershell
python scripts/select_enrollment_sentences.py ^
  --input data/enrollment_candidates/candidates.txt ^
  --method phoneme ^
  --n 5 ^
  --output results/phoneme_selected_sentences.csv ^
  --write-app-config
```

Restart Streamlit. The Enrollment tab will display the selected five sentences.

Generate the random control:

```powershell
python scripts/select_enrollment_sentences.py ^
  --input data/enrollment_candidates/candidates.txt ^
  --method random ^
  --n 5 ^
  --seed 42 ^
  --output results/random_selected_sentences.csv
```

After recording both five-sentence conditions for several speakers, fill the two
manifest CSVs in `results/enrollment_experiment/` and run:

```powershell
python training/evaluate_enrollment_methods.py ^
  --enrollment-csv results/enrollment_experiment/enrollment_samples.csv ^
  --query-csv results/enrollment_experiment/query_samples.csv ^
  --output-csv results/enrollment_experiment/random_vs_phoneme.csv
```

## Stage 13 — Optional noise smoke test

```powershell
python scripts/add_test_noise.py ^
  --input user1_query.wav ^
  --output data/experiments/user1_query_10db.wav ^
  --snr-db 10
```

Then run SV/SID on the noisy WAV and compare score/result with the clean one.

## Stage 14 — Package submission

When all checks are done:

```powershell
python scripts/package_submission.py --student-ids STUDENT_ID_1 STUDENT_ID_2 --include-model
```

If the model is submitted by a separate Drive/Hugging Face link:

```powershell
python scripts/package_submission.py --student-ids STUDENT_ID_1 STUDENT_ID_2
```

The ZIP is created under `submission/`.
