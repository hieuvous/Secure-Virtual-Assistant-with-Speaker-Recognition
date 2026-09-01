import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.database.db import init_db
from src.database.repositories import upsert_profile
from src.speaker.model import get_ecapa
from src.speaker.profile import create_speaker_profile

p=argparse.ArgumentParser()
p.add_argument("--user-id",type=int,required=True)
p.add_argument("--audio",nargs="+",required=True)
a=p.parse_args()
if len(a.audio)!=5:
    print("WARNING: final evaluation protocol used 5 enrollment recordings.")
init_db()
r=create_speaker_profile(a.user_id,a.audio)
s=get_ecapa()
v="finetuned_epoch10" if s.using_finetuned else "pretrained_voxceleb"
upsert_profile(a.user_id,r["embedding"],r["num_samples"],v)
print(r);print("model_version:",v)
