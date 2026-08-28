# Run & Test Guide

## A. Local VS Code

Recommended: Python 3.10.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## B. Test 0 — validate the uploaded release WITHOUT downloading Hugging Face models

```powershell
python scripts/validate_release_artifacts.py
```

Expected:
- PASS
- epoch 10
- embedding dim 192
- 600 training speakers
- SV threshold 0.1566438227891922

## C. Test 1 — unit tests

```powershell
pytest -q tests/test_speaker_logic.py tests/test_router.py tests/test_scoring.py
```

## D. Test 2 — first real ECAPA/VAD run

Internet is needed the first time to cache:
- `speechbrain/spkrec-ecapa-voxceleb`
- `speechbrain/vad-crdnn-libriparty`

After they are cached, the demo can run offline.

```powershell
python scripts/smoke_test_ecapa.py path\to\one.wav
```

Then:

```powershell
python scripts/init_db.py
streamlit run app/main.py
```

## E. Enrollment smoke test

Create user(s), then record 5 utterances in Streamlit, or:

```powershell
python scripts/enroll_from_files.py --user-id 1 --audio e1.wav e2.wav e3.wav e4.wav e5.wav
```

Enroll at least 2–3 users.

## F. SV tests

Correct speaker:

```powershell
python scripts/test_sv.py --user-id 1 --audio user1_query.wav
```

Wrong speaker claiming user 1:

```powershell
python scripts/test_sv.py --user-id 1 --audio other_person.wav
```

Expected:
- genuine should usually pass;
- impostor should usually reject;
- actual score must be inspected, not assumed.

## G. SID smoke test

```powershell
python scripts/test_sid.py --audio user1_query.wav
python scripts/test_sid.py --audio unknown_person.wav
```

Until SID calibration is run, output contains:
`sid_threshold_calibrated=false`.

## H. REQUIRED before declaring Đợt 3 complete — calibrate SID threshold

Run on Colab/Kaggle using the speaker-disjoint DEV CSV:

```bash
python training/evaluate_identification.py \
  --dev-csv /content/SpeakerRecognition/metadata_v2/dev.csv \
  --checkpoint models/ecapa_vietnamceleb_epoch10.pt \
  --output-json sid_dev_metrics.json \
  --gallery-speakers 20 \
  --unknown-speakers 10 \
  --enroll-utts 5 \
  --seed 42
```

Copy `sid_dev_metrics.json` to local and run:

```powershell
python scripts/update_sid_threshold.py --sid-json sid_dev_metrics.json
```

Then rerun SID known + unknown tests.

## I. Fine-tuning / report note

Use `notebooks/finetune_ecapa_colab.ipynb` as the final experiment notebook.

For the report, use ONLY the final all-impostor release values in:
- `results/all_impostor_metrics.json`
- `results/pretrained_vs_finetuned.csv`

Do not quote older exploratory EER values from notebook history.
