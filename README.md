# Secure Student Virtual Assistant — Starter

Starter code for the **Secure Virtual Assistant with Speaker Recognition** final project.

## What this ZIP already contains

- Local development structure for **VS Code + Python 3.10**
- SpeechBrain **ECAPA-TDNN pretrained** embedding extraction
- Speaker enrollment: multiple recordings → normalized mean embedding
- Speaker Verification (1:1 cosine similarity)
- Speaker Identification (1:N cosine similarity + unknown rejection)
- Faster-Whisper ASR wrapper
- Streamlit microphone UI using `st.audio_input`
- SQLite schema + basic repositories
- Rule-based intent router + permission map
- Kaggle-oriented Vietnam-Celeb subset preparation
- A practical ECAPA fine-tuning script
- Verification evaluation script to choose threshold from development data

This is a **starter**, not the final submission. TTS, phoneme-based enrollment optimization,
full experiment tables, final report figures, and stronger tests are intentionally left for later iterations.

---

# 1. Recommended workflow

Use two environments:

1. **Local / VS Code on Windows**
   - Streamlit application
   - pretrained/fine-tuned ECAPA inference
   - Faster-Whisper ASR
   - SQLite
   - enrollment + demo

2. **Kaggle GPU**
   - prepare/check Vietnam-Celeb subset
   - fine-tune ECAPA
   - evaluate
   - export `finetuned_ecapa.pt`
   - copy checkpoint back to local `models/`

The local app automatically uses `models/finetuned_ecapa.pt` if that file exists;
otherwise it uses the pretrained VoxCeleb ECAPA.

---

# 2. Python version

Recommended: **Python 3.10**

On Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code:
- Open this project folder.
- `Ctrl+Shift+P`
- `Python: Select Interpreter`
- choose `.venv\Scripts\python.exe`.

---

# 3. Initialize the project

```powershell
python scripts/init_db.py
python scripts/seed_demo_data.py
```

Smoke-test ECAPA:

```powershell
python scripts/smoke_test_ecapa.py path\to\your_audio.wav
```

The first run downloads the pretrained model:
`speechbrain/spkrec-ecapa-voxceleb`.

---

# 4. Run Streamlit locally

```powershell
streamlit run app/main.py
```

Current starter pages:

- **Assistant**
  - record Vietnamese command
  - ASR
  - detect intent
  - GENERAL / IDENTIFICATION / VERIFICATION access control
  - execute basic actions

- **Enrollment**
  - create/select a user
  - record 5 enrollment utterances
  - ECAPA embedding for each utterance
  - normalized mean embedding
  - save speaker profile

The current thresholds in `configs/thresholds.json` are **PROVISIONAL ONLY**.
Replace them after running development-set evaluation.

---

# 5. Dataset: Vietnam-Celeb

Use the official project repository:

https://github.com/thanhpv2102/Vietnam-Celeb.Interspeech

The official repository provides four dataset parts and these important files:

- `vietnam-celeb-t.txt` — training utterance list
- `vietnam-celeb-e.txt` — Vietnam-Celeb-E verification trials
- `vietnam-celeb-h.txt` — Vietnam-Celeb-H verification trials
- speaker metadata TSV

For this 9-day project, start with a subset rather than all 880 training speakers.

Recommended first run:

- 150 speakers for fine-tuning
- maximum 20 utterances per training speaker
- 30 additional speaker-disjoint speakers for development / threshold selection

Do **not** use the final test trials to choose thresholds.

---

# 6. How to organize Vietnam-Celeb for the provided scripts

Expected extracted structure:

```text
/kaggle/input/vietnam-celeb/
├── data/
│   ├── speaker_001/
│   │   ├── *.wav
│   │   └── ...
│   ├── speaker_002/
│   └── ...
├── vietnam-celeb-t.txt
├── vietnam-celeb-e.txt
└── vietnam-celeb-h.txt
```

The exact speaker IDs can be numeric; the scripts do not require the example names above.

If Kaggle input path is different, just change the command arguments.

---

# 7. Kaggle: prepare subset

Create a Kaggle Notebook with GPU enabled, upload this project ZIP or add it as a Kaggle Dataset,
then run a terminal cell or Python shell command equivalent to:

```bash
python training/prepare_vietnam_celeb_subset.py \
  --data-root /kaggle/input/vietnam-celeb/data \
  --official-train-list /kaggle/input/vietnam-celeb/vietnam-celeb-t.txt \
  --output-dir /kaggle/working/metadata \
  --train-speakers 150 \
  --dev-speakers 30 \
  --max-utts 20 \
  --seed 42
```

Outputs:

```text
/kaggle/working/metadata/
├── train.csv
├── val.csv
├── dev.csv
└── split_summary.json
```

`dev.csv` is speaker-disjoint from the fine-tuning speakers.

---

# 8. Kaggle: install training dependencies

Kaggle usually already contains PyTorch. Install only the project dependencies needed there:

```bash
pip install -q speechbrain==1.1.0 pandas scikit-learn soundfile
```

Then check:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

---

# 9. Fine-tune ECAPA on Kaggle

First do a short sanity run:

```bash
python training/finetune_ecapa_kaggle.py \
  --train-csv /kaggle/working/metadata/train.csv \
  --val-csv /kaggle/working/metadata/val.csv \
  --output /kaggle/working/finetuned_ecapa.pt \
  --epochs 1 \
  --batch-size 16 \
  --lr 1e-4
```

If that works, run the real first experiment:

```bash
python training/finetune_ecapa_kaggle.py \
  --train-csv /kaggle/working/metadata/train.csv \
  --val-csv /kaggle/working/metadata/val.csv \
  --output /kaggle/working/finetuned_ecapa.pt \
  --epochs 5 \
  --batch-size 16 \
  --lr 1e-4
```

Notes:
- This starter fine-tunes the pretrained ECAPA embedding network using a new speaker-classification head.
- The classification head uses an AAM-Softmax-style angular margin objective.
- `1e-4`, 5 epochs, batch 16 are **project starting values**, not claimed as the original Vietnam-Celeb paper hyperparameters.
- If GPU memory is comfortable, try batch size 32.
- Keep the pretrained baseline; do not overwrite it.

---

# 10. Choose the SV threshold on development speakers

After training:

```bash
python training/evaluate_verification.py \
  --dev-csv /kaggle/working/metadata/dev.csv \
  --checkpoint /kaggle/working/finetuned_ecapa.pt \
  --output-json /kaggle/working/sv_dev_metrics.json \
  --max-utts-per-speaker 8 \
  --seed 42
```

The JSON contains:

- EER
- FAR at the selected threshold
- FRR at the selected threshold
- selected threshold
- number of positive/negative trials

Copy the chosen value into:

```text
configs/thresholds.json
```

Do this separately for SID later; do not assume the same threshold is optimal.

---

# 11. Bring the fine-tuned model back to VS Code

Download from Kaggle:

```text
finetuned_ecapa.pt
```

Place it here:

```text
models/finetuned_ecapa.pt
```

The local `ECAPAService` detects this file automatically.

Then rerun:

```powershell
streamlit run app/main.py
```

---

# 12. Important folders

```text
app/                    Streamlit UI
src/speaker/            ECAPA, enrollment, SV, SID
src/speech/             audio preprocessing, Faster-Whisper
src/assistant/          intents, permissions, actions
src/database/           SQLite
training/               subset, fine-tune, evaluation
configs/                thresholds and model settings
scripts/                setup and smoke tests
models/                 local fine-tuned checkpoint (not Git)
data/                   runtime/user data (not Git)
tests/                   lightweight unit tests
```

---

# 13. First things to test

In this exact order:

1. `python scripts/init_db.py`
2. `python scripts/smoke_test_ecapa.py sample.wav`
3. `streamlit run app/main.py`
4. create two users
5. enroll both users
6. check SID distinguishes them
7. check wrong speaker fails SV
8. check Faster-Whisper returns Vietnamese transcript
9. prepare Vietnam-Celeb subset on Kaggle
10. run one-epoch fine-tuning sanity test

Do not start phoneme-based enrollment optimization until the above path works end-to-end.
