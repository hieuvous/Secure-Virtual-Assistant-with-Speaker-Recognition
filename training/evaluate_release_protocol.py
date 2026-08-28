"""Evaluate the released SV protocol: VAD + 5-recording centroid + all impostors."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.VAD import VAD

SR = 16000

def parse():
    p=argparse.ArgumentParser()
    p.add_argument("--dev-csv", required=True)
    p.add_argument("--test-csv")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--enroll-utts", type=int, default=5)
    return p.parse_args()

def load16(path):
    w,sr=torchaudio.load(path)
    if w.shape[0]>1: w=w.mean(0,keepdim=True)
    if sr!=SR: w=torchaudio.functional.resample(w,sr,SR)
    return w.float()

def main():
    a=parse()
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    model=EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/eval_base", run_opts={"device":device})
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    model.mods.embedding_model.load_state_dict(ck["embedding_model"],strict=True)
    model.mods.embedding_model.eval()
    vad=VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty",
                         savedir="models/eval_vad",run_opts={"device":device})

    def e(path):
        w=load16(path)
        try:
            b=vad.get_speech_segments(path)
            seg=[]
            for x in b:
                s0,e0=int(float(x[0])*SR),int(float(x[1])*SR)
                if e0>s0: seg.append(w[:,s0:e0])
            if seg: w=torch.cat(seg,dim=1)
        except Exception: pass
        with torch.inference_mode():
            x=model.encode_batch(w.to(device),normalize=False).squeeze().cpu()
        return F.normalize(x,p=2,dim=0)

    def build(csv):
        df=pd.read_csv(csv); profiles={}; queries={}
        for spk,g in df.groupby("speaker_id"):
            paths=sorted(g["path"].astype(str).tolist())
            if len(paths)<=a.enroll_utts: continue
            ens, qs=paths[:a.enroll_utts],paths[a.enroll_utts:]
            c=torch.stack([e(p) for p in ens]).mean(0)
            profiles[str(spk)]=F.normalize(c,p=2,dim=0)
            queries[str(spk)]=qs
        return profiles,queries

    def score(profiles,queries):
        scores=[];labels=[]
        spks=sorted(profiles)
        for true in spks:
            for q in queries[true]:
                qe=e(q)
                for cand in spks:
                    scores.append(F.cosine_similarity(qe,profiles[cand],dim=0).item())
                    labels.append(1 if cand==true else 0)
        scores=np.asarray(scores);labels=np.asarray(labels)
        fpr,tpr,thr=roc_curve(labels,scores,pos_label=1);fnr=1-tpr
        i=int(np.nanargmin(np.abs(fpr-fnr)))
        return float((fpr[i]+fnr[i])/2),float(thr[i]),scores,labels

    dp,dq=build(a.dev_csv)
    de,dt,_,_=score(dp,dq)
    out={"dev_eer":de,"dev_threshold":dt,"dev_speakers":len(dp),
         "enrollment_utts_per_speaker":a.enroll_utts}

    if a.test_csv:
        tp,tq=build(a.test_csv)
        te,_,s,l=score(tp,tq)
        far=float(np.mean(s[l==0]>=dt));frr=float(np.mean(s[l==1]<dt))
        out.update({"test_eer":te,"test_far_at_dev_threshold":far,
                    "test_frr_at_dev_threshold":frr,"test_speakers":len(tp),
                    "test_genuine_trials":int((l==1).sum()),
                    "test_impostor_trials":int((l==0).sum())})

    Path(a.output_json).write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
