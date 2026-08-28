# Changelog — Đợt 5 + Đợt 6

## Added

### Experiment / Đợt 5
- `requirements-experiments.txt`
- `src/speaker/enrollment_selection.py`
- `scripts/select_enrollment_sentences.py`
- `training/evaluate_enrollment_methods.py`
- `scripts/add_test_noise.py`
- `data/enrollment_candidates/README.md`
- `results/enrollment_experiment/enrollment_samples_template.csv`
- `results/enrollment_experiment/query_samples_template.csv`

### Final / Đợt 6
- `scripts/final_system_check.py`
- `scripts/package_submission.py`
- `tests/test_actions.py`
- `tests/test_enrollment_selection.py`
- `DOT5_DOT6_GUIDE.md`
- `FINAL_RUN_GUIDE.md`
- `FINAL_CHECKLIST.md`
- `docs/ARCHITECTURE.md`
- `docs/DEMO_SCRIPT.md`
- `docs/REPORT_OUTLINE.md`
- `docs/EXPERIMENT_TABLES.md`
- `CHANGELOG_DOT5_DOT6.md`

## Updated
- `app/main.py`
  - Model/Evaluation tab;
  - display final SV metrics;
  - display chosen enrollment sentences;
  - save enrollment method.

- `src/assistant/router.py`
  - deterministic quoted-title extraction;
  - simplified to final 8 demo intents.

- `src/assistant/permissions.py`
  - final General / SID / SV permission mapping.

- `src/assistant/actions.py`
  - implements course-room lookup;
  - schedule lookup;
  - task add/delete;
  - private-note read.

- `src/database/repositories.py`
  - new schedule/course/task helpers;
  - audit-log query.

- `src/database/schema.sql`
  - adds `course_rooms`.

- `scripts/seed_demo_data.py`
  - seeds course room + task + private note + schedule.

## Removed
- `.pytest_cache/` from the distributed ZIP only.
- No required source/model/result files were removed.

## Still intentionally optional
- phoneme-based enrollment experiment;
- noise experiment;
- TTS;
- fancy evaluation dashboard.

## Still required before claiming full SID completion
If `configs/thresholds.json` still says `sid_status=NEEDS_CALIBRATION`, run
`training/evaluate_identification.py`, then `scripts/update_sid_threshold.py`.
