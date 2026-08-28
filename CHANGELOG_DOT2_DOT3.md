# Changes from Đợt 1 → Final Đợt 2 + Đợt 3

## Đợt 2 — added/replaced

### Added
- `notebooks/EDA.ipynb` — patched to final 600-speaker configuration.
- `notebooks/finetune_ecapa_colab.ipynb` — cleaned copy of the final notebook.
- `models/ecapa_vietnamceleb_epoch10.pt` — released Epoch-10 checkpoint.
- `models/config.json`
- `results/all_impostor_metrics.json`
- `results/pretrained_vs_finetuned.csv`
- `training/finetune_ecapa_colab.py` — consolidated rerun-compatible trainer.
- `training/evaluate_release_protocol.py`

### Replaced
- `training/prepare_vietnam_celeb_subset.py`
  - default 600 train + 50 dev speakers;
  - restricts files using the official Vietnam-Celeb-T list;
  - speaker-disjoint split.

### Final model facts verified from the actual checkpoint
- epoch = 10
- embedding_dim = 192
- training speaker classes = 600
- state_dict strictly matches SpeechBrain ECAPA-TDNN

### Final authoritative SV result
- Pretrained: DEV EER 13.30%, TEST EER 11.91%.
- Fine-tuned Epoch 10: DEV EER 9.98%, TEST EER 8.42%.
- Fine-tuned TEST FAR at locked DEV threshold: 9.00%.
- Fine-tuned TEST FRR at locked DEV threshold: 7.88%.
- SV threshold = 0.1566438227891922.
- TEST: 50 unseen speakers, 698 genuine + 34,202 impostor trials, speaker overlap 0.

## Đợt 3 — replaced/added

### Replaced
- `src/speaker/model.py`
- `src/speaker/embedding.py`
- `src/speaker/profile.py`
- `src/speaker/scoring.py`
- `src/speaker/verification.py`
- `src/speaker/identification.py`
- `configs/settings.yaml`
- `configs/thresholds.json`

### Added
- `src/speech/vad.py`
- `training/evaluate_identification.py`
- `scripts/validate_release_artifacts.py`
- `scripts/enroll_from_files.py`
- `scripts/test_sv.py`
- `scripts/test_sid.py`
- `scripts/update_sid_threshold.py`
- `tests/test_speaker_logic.py`

## Important correction
The final SV threshold was calibrated with:
`VAD → ECAPA → L2 → 5-recording mean profile → cosine → threshold`.

Therefore local Enrollment/SV/SID now uses VAD too. Reusing `0.1566` without VAD would be a protocol mismatch.

## Still pending
SID threshold is not present in the released SV artifacts. It must be calibrated separately with `training/evaluate_identification.py`.
