# Report outline

## 1. Introduction
Problem, security/personalization motivation, project objectives.

## 2. Background
Speaker Verification, Speaker Identification, embeddings, cosine similarity,
ECAPA-TDNN, enrollment.

## 3. Dataset
Vietnam-Celeb, selected subset, speaker-disjoint protocol, train/validation/dev/test.

## 4. Model and fine-tuning
Base SpeechBrain ECAPA, transfer learning, training configuration, Epoch 10 selection.

## 5. Verification evaluation
Enrollment = 5 utterances, VAD, mean L2 profile, all-impostor protocol,
EER/FAR/FRR, DEV threshold locked before TEST.

## 6. Identification
1:N ranking, separate SID threshold, unknown rejection, evaluation results.

## 7. Enrollment optimization
Only if completed:
candidate corpus → Vietnamese G2P → greedy phoneme coverage →
random-vs-phoneme comparison.
Clearly state that the greedy implementation is the group's implementation proposal
unless the exact paper algorithm is verified from the full paper.

## 8. System architecture
ASR, NLU, access control, SV/SID, SQLite, Streamlit.

## 9. Application functions
General / personalized / sensitive use cases.

## 10. Results
Use final release metrics only.
Do not mix old exploratory notebook EER numbers.

## 11. Demo and testing
registered, wrong, unknown, noise, ASR errors, duplicate enrollment.

## 12. Limitations
No anti-spoofing/replay defense, subset training, rule-based NLU,
optional enrollment experiment scale.

## 13. Future work
Anti-spoofing, larger training set, better open-set calibration,
more rigorous phoneme-selection experiment.

## 14. Conclusion
Summarize model improvement and end-to-end integration.
