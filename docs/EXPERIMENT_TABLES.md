# Experiment tables

## A. Pretrained vs fine-tuned — REQUIRED

| Model | DEV EER | TEST EER | TEST FAR @ DEV thr | TEST FRR @ DEV thr |
|---|---:|---:|---:|---:|
| Pretrained ECAPA | 13.30% | 11.91% | 9.46% | 13.47% |
| Fine-tuned Epoch 10 | 9.98% | 8.42% | 9.00% | 7.88% |

## B. Protocol — REQUIRED

| Item | Value |
|---|---:|
| DEV speakers | 37 |
| TEST speakers | 50 |
| Enrollment utterances/speaker | 5 |
| TEST genuine trials | 698 |
| TEST impostor trials | 34,202 |
| Speaker overlap | 0 |
| VAD | speechbrain/vad-crdnn-libriparty |

## C. SID — REQUIRED before final SID claim

| Model | Gallery speakers | Enroll utts | Closed-set acc | Known open-set acc | Unknown rejection | SID threshold |
|---|---:|---:|---:|---:|---:|---:|
| Fine-tuned Epoch 10 | | 5 | | | | |

## D. Enrollment optimization — OPTIONAL

| Method | # speakers | N | SID acc | SV EER |
|---|---:|---:|---:|---:|
| Random | | 5 | | |
| Phoneme coverage | | 5 | | |

## E. Noise — OPTIONAL

| Audio | SNR | SV score/result | SID result |
|---|---:|---|---|
| Clean | — | | |
| Noisy | 20 dB | | |
| Noisy | 10 dB | | |
