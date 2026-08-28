# Final architecture

```text
Microphone / WAV
      │
      ├──────────────→ Faster-Whisper → transcript → rule-based intent/entities
      │                                      │
      │                                      ▼
      │                              Permission mapping
      │                        GENERAL / SID / VERIFICATION
      │                                      │
      ▼                                      │
SpeechBrain VAD                              │
      │                                      │
      ▼                                      │
Fine-tuned ECAPA-TDNN (192-D)                │
      │                                      │
      ├── SID 1:N → user / UNKNOWN ──────────┤
      │                                      │
      └── SV 1:1 → accept / reject ──────────┤
                                             ▼
                                      Access Control
                                             │
                                             ▼
                                      SQLite action
                                             │
                                             ▼
                                      Text response
```

## Speaker enrollment

```text
5 recordings
→ VAD per recording
→ ECAPA embedding
→ L2 normalization
→ mean centroid
→ L2 normalization
→ speaker profile
```

## Important design decision

SV and SID use the same ECAPA embedding model but different decision protocols:
- SV compares query with one claimed identity.
- SID ranks query against all enrolled profiles and applies a separate unknown threshold.
