# Secure Student Virtual Assistant with Speaker Recognition

Final cumulative project package.

## Core model
- ECAPA-TDNN / SpeechBrain
- Vietnam-Celeb fine-tuning
- selected checkpoint: Epoch 10
- 192-D embeddings
- VAD: `speechbrain/vad-crdnn-libriparty`
- enrollment: five recordings → mean L2-normalized profile
- cosine similarity

## Released SV metrics
- Pretrained DEV EER: 13.30%
- Fine-tuned DEV EER: 9.98%
- Pretrained TEST EER: 11.91%
- Fine-tuned TEST EER: 8.42%
- SV threshold: 0.1566438227891922 (DEV all-impostor)
- TEST speakers: 50 unseen
- speaker overlap: 0

## Application
- Streamlit voice interface
- Faster-Whisper Vietnamese ASR
- rule-based intent/entity routing
- SQLite
- General actions
- SID-personalized actions
- SV-protected sensitive actions
- enrollment/user management
- audit logging

## Start here
Read:
1. `FINAL_RUN_GUIDE.md`
2. `FINAL_CHECKLIST.md`
3. `DOT5_DOT6_GUIDE.md`

Then run:

```powershell
python scripts/final_system_check.py
pytest -q
python scripts/validate_release_artifacts.py
```

Real model test:

```powershell
python scripts/smoke_test_ecapa.py sample.wav
```

Application:

```powershell
streamlit run app/main.py
```

## Optional experiment
`requirements-experiments.txt` is only for the phoneme-based enrollment experiment.
It is not required to run the core application.
