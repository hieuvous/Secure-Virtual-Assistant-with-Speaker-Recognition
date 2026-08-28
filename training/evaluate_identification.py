"""Calibrate SID unknown-rejection threshold separately from the SV threshold."""

from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torchaudio
import torch.nn.functional as F
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.VAD import VAD

SR=16000

def norm(x): return x/max(float(torch.linalg.vector_norm(x)),1e-12)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dev-csv",required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output-json",required=True)
    p.add_argument("--gallery-speakers",type=int,default=20)
    p.add_argument("--unknown-speakers",type=int,default=10)
    p.add_argument("--enroll-utts",type=int,default=5)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); rng=random.Random(a.seed)
    device="cuda:0" if torch.cuda.is_available() else "cpu"

    model=EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/sid_base",run_opts={"device":device})
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    model.mods.embedding_model.load_state_dict(ck["embedding_model"],strict=True)
    vad=VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty",
                        savedir="models/sid_vad",run_opts={"device":device})

    def emb(path):
        w,sr=torchaudio.load(path)
        if w.shape[0]>1:w=w.mean(0,keepdim=True)
        if sr!=SR:w=torchaudio.functional.resample(w,sr,SR)
        try:
            b=vad.get_speech_segments(path); pieces=[]
            for x in b:
                s0,e0=int(float(x[0])*SR),int(float(x[1])*SR)
                if e0>s0:pieces.append(w[:,s0:e0])
            if pieces:w=torch.cat(pieces,dim=1)
        except Exception:pass
        with torch.inference_mode():
            x=model.encode_batch(w.to(device),normalize=False).squeeze().cpu()
        return norm(x)

    df=pd.read_csv(a.dev_csv); groups={}
    for s,g in df.groupby("speaker_id"):
        paths=sorted(g["path"].astype(str).tolist())
        if len(paths)>=a.enroll_utts+1: groups[str(s)]=paths
    spks=list(groups);rng.shuffle(spks)
    need=a.gallery_speakers+a.unknown_speakers
    if len(spks)<need:raise RuntimeError(f"Need {need} eligible speakers, found {len(spks)}")
    gallery_spks=spks[:a.gallery_speakers]
    unknown_spks=spks[a.gallery_speakers:need]

    cache={}
    def E(p):
        if p not in cache:cache[p]=emb(p)
        return cache[p]

    gallery={}
    for s in gallery_spks:
        gallery[s]=norm(torch.stack([E(x) for x in groups[s][:a.enroll_utts]]).mean(0))

    known=[]
    for s in gallery_spks:
        for q in groups[s][a.enroll_utts:]:
            qe=E(q)
            ranked=sorted(((c,float(F.cosine_similarity(qe,v,dim=0))) for c,v in gallery.items()),
                          key=lambda z:z[1],reverse=True)
            known.append({"correct":ranked[0][0]==s,"score":ranked[0][1]})

    unknown=[]
    for s in unknown_spks:
        for q in groups[s]:
            qe=E(q)
            unknown.append(max(float(F.cosine_similarity(qe,v,dim=0)) for v in gallery.values()))

    vals=np.unique(np.asarray([r["score"] for r in known]+unknown))
    closed=sum(r["correct"] for r in known)/len(known)
    best=None
    for t in vals:
        known_open=sum(r["correct"] and r["score"]>=t for r in known)/len(known)
        unknown_rej=sum(x<t for x in unknown)/len(unknown)
        balanced=(known_open+unknown_rej)/2
        row=(balanced,known_open,unknown_rej,float(t))
        if best is None or row>best:best=row

    out={"sid_threshold":best[3],"closed_set_accuracy":closed,
         "known_open_set_accuracy":best[1],"unknown_rejection_rate":best[2],
         "balanced_score":best[0],"gallery_speakers":len(gallery_spks),
         "unknown_speakers":len(unknown_spks),"enroll_utts_per_speaker":a.enroll_utts,
         "known_queries":len(known),"unknown_queries":len(unknown),"seed":a.seed,
         "threshold_selection":"DEV; maximize mean(known open-set accuracy, unknown rejection rate)"}
    Path(a.output_json).write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
