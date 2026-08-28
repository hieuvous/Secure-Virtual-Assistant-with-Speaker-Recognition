# Enrollment candidate sentences

This folder intentionally contains **no invented five-sentence set**.

For the phoneme-based experiment, prepare a candidate pool from a legitimate
Vietnamese transcript corpus (for example, the transcript portion of a corpus
you are allowed to use), then save one of:

```text
data/enrollment_candidates/candidates.txt
```

one sentence per line, or:

```csv
text
...
```

Recommended cleanup before selection:
- normal Vietnamese declarative sentences;
- no extremely short one/two-word lines;
- remove corrupt symbols/URLs;
- keep a few hundred candidates if possible.

Then run:

```powershell
pip install -r requirements-experiments.txt
python scripts/select_enrollment_sentences.py ^
  --input data/enrollment_candidates/candidates.txt ^
  --method phoneme ^
  --n 5 ^
  --output results/phoneme_selected_sentences.csv ^
  --write-app-config
```

For the random control:

```powershell
python scripts/select_enrollment_sentences.py ^
  --input data/enrollment_candidates/candidates.txt ^
  --method random ^
  --n 5 ^
  --seed 42 ^
  --output results/random_selected_sentences.csv
```
