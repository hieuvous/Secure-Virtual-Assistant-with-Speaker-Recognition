# Kaggle checklist

## Before training
- Enable GPU in Kaggle Notebook settings.
- Add/expose the extracted Vietnam-Celeb dataset as a Kaggle input.
- Upload this project ZIP or add the source folder as a Kaggle Dataset.
- Confirm the folder containing speaker subfolders.

## 1. Prepare metadata
Run `training/prepare_vietnam_celeb_subset.py`.

Recommended first configuration:
- 150 fine-tuning speakers
- 30 speaker-disjoint development speakers
- max 20 utterances/speaker
- 10% utterances from fine-tuning speakers for classification validation
- random seed 42

## 2. Sanity train
Run only 1 epoch first.
Do not start a long run until:
- train loss is finite,
- checkpoint is saved,
- GPU is actually being used.

## 3. First real fine-tune
Start with:
- 5 epochs
- batch 16
- LR 1e-4
- 3-second chunks
- AAM margin 0.2
- scale 30

These are project starting values. Keep them clearly separated from paper/recipe hyperparameters in the report.

## 4. Evaluate threshold
Use the speaker-disjoint `dev.csv`.
Run both:
- pretrained baseline (omit `--checkpoint`)
- fine-tuned model (pass `--checkpoint`)

Never choose the final threshold on Vietnam-Celeb-E/H.

## 5. Export
Download:
- `finetuned_ecapa.pt`
- `finetuned_ecapa.history.json`
- `sv_dev_metrics.json`
- metadata CSVs + `split_summary.json`

Put only the checkpoint in `models/` locally. Do not commit the checkpoint to Git.
