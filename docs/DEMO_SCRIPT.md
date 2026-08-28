# 5–7 minute demo

## 0:00–0:40 — Goal
Explain:
- ECAPA is the technical core.
- SV protects sensitive functions.
- SID personalizes responses.

## 0:40–1:40 — Enrollment
Create/enroll one user with five recordings.
Show:
- five recordings;
- 192-D profile;
- model = fine-tuned Epoch 10.

## 1:40–2:15 — General
Say:
`Bây giờ là mấy giờ?`

Show:
- intent GET_TIME;
- auth GENERAL;
- no speaker authentication.

## 2:15–3:10 — Personalized/SID
Say:
`Tôi còn deadline nào?`

Show:
- SID best speaker;
- score/threshold;
- user's own tasks.

## 3:10–4:00 — Sensitive/SV success
Choose claimed user.
Say:
`Đọc ghi chú riêng của tôi`

Show:
- VERIFICATION required;
- cosine score;
- accepted;
- private note returned.

## 4:00–4:50 — Sensitive/SV failure
Another person speaks while claimed user is unchanged.

Show:
- score below threshold;
- action rejected;
- private data not returned.

## 4:50–5:30 — Unknown SID
Unregistered person asks for tasks.

Show UNKNOWN/rejection if SID calibration has been completed.

## 5:30–6:30 — Experiment
Show:
- pretrained TEST EER 11.91%;
- fine-tuned TEST EER 8.42%;
- DEV-calibrated SV threshold 0.1566;
- optional random-vs-phoneme result if completed.
