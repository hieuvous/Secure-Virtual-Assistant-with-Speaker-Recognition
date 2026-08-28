"""Offline structural validation: no Hugging Face download required."""
from pathlib import Path
import json, torch, pandas as pd
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

ROOT=Path(__file__).resolve().parents[1]
ck=torch.load(ROOT/"models/ecapa_vietnamceleb_epoch10.pt",map_location="cpu",weights_only=False)
assert ck["epoch"]==10
assert ck["embedding_dim"]==192
assert len(ck["label_map"])==600

model=ECAPA_TDNN(
    input_size=80,
    channels=[1024,1024,1024,1024,3072],
    kernel_sizes=[5,3,3,3,1],
    dilations=[1,2,3,4,1],
    attention_channels=128,
    lin_neurons=192,
)
model.load_state_dict(ck["embedding_model"],strict=True)

config=json.loads((ROOT/"models/config.json").read_text())
metrics=json.loads((ROOT/"results/all_impostor_metrics.json").read_text())
csv=pd.read_csv(ROOT/"results/pretrained_vs_finetuned.csv")

assert abs(config["verification_threshold"]-metrics["fine_tuned_epoch_10"]["dev_threshold"])<1e-12
assert metrics["protocol"]["speaker_overlap"]==0
assert len(csv)==2

print("PASS: checkpoint format + strict ECAPA load + release config/metrics are consistent")
print("epoch:",ck["epoch"])
print("embedding_dim:",ck["embedding_dim"])
print("training_speakers:",len(ck["label_map"]))
print("SV threshold:",config["verification_threshold"])
print("Fine-tuned DEV EER:",metrics["fine_tuned_epoch_10"]["dev_eer"])
print("Fine-tuned TEST EER:",metrics["fine_tuned_epoch_10"]["test_eer"])
