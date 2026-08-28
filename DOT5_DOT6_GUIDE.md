# ĐỢT 5 + ĐỢT 6 — Experiments + Finalization

This ZIP is **cumulative**: it already contains the merged Đợt 1–3 project and
adds the Đợt 5/6 changes. You do not need to copy patches manually.

## What Đợt 5 adds

1. Optional phoneme-based sentence selection:
   - `src/speaker/enrollment_selection.py`
   - `scripts/select_enrollment_sentences.py`
   - `requirements-experiments.txt`
   - `data/enrollment_candidates/README.md`

2. Random vs phoneme enrollment experiment:
   - `training/evaluate_enrollment_methods.py`
   - manifest templates under `results/enrollment_experiment/`

3. Simple optional noise test helper:
   - `scripts/add_test_noise.py`

## What Đợt 6 updates

1. Completes the demo DB actions that were still placeholders:
   - `src/assistant/actions.py`
   - `src/assistant/router.py`
   - `src/assistant/permissions.py`
   - `src/database/repositories.py`
   - `src/database/schema.sql`

2. Updates Streamlit:
   - `app/main.py`
   - adds Model/Evaluation tab;
   - shows optimized enrollment sentences if generated;
   - stores enrollment method.

3. Adds final testing and packaging:
   - `scripts/final_system_check.py`
   - `scripts/package_submission.py`
   - `tests/test_actions.py`
   - `tests/test_enrollment_selection.py`

4. Adds final documentation:
   - `FINAL_RUN_GUIDE.md`
   - `FINAL_CHECKLIST.md`
   - `docs/ARCHITECTURE.md`
   - `docs/DEMO_SCRIPT.md`
   - `docs/REPORT_OUTLINE.md`
   - `docs/EXPERIMENT_TABLES.md`
   - `CHANGELOG_DOT5_DOT6.md`

## Files removed

Only runtime/cache files such as `.pytest_cache/` were removed.
No source/model/result artifact from Đợt 1–3 was intentionally removed.

## Do I need another application?

### Required: NO

Core project uses:
- VS Code or any Python IDE;
- Python 3.10;
- the existing Python packages;
- internet only on the first model-cache download.

You do NOT need Docker, FastAPI, React, a vector database, or another desktop application.

### Optional phoneme experiment

The phoneme experiment uses Python package `viphoneme` plus its Python dependencies.
Install only if you do that experiment:

```powershell
pip install -r requirements-experiments.txt
```

Viphoneme is isolated from the core assistant; if it fails to install, SV/SID/app are unaffected.

## Core commands to run

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/final_system_check.py
pytest -q
python scripts/validate_release_artifacts.py

python scripts/init_db.py
python scripts/seed_demo_data.py
python scripts/smoke_test_ecapa.py path\to\sample.wav

streamlit run app/main.py
```

Then perform real enrollment/SV/SID tests described in `FINAL_RUN_GUIDE.md`.
