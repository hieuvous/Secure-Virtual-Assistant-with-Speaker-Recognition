import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.database.db import init_db
from src.database.repositories import list_profiles
from src.speaker.identification import identify_speaker
p=argparse.ArgumentParser();p.add_argument("--audio",required=True);a=p.parse_args()
init_db();profiles=list_profiles()
if not profiles:raise RuntimeError("Enroll at least 2-3 users first.")
r=identify_speaker(a.audio,profiles)
print(json.dumps(r,indent=2))
if not r.get("sid_threshold_calibrated",False):
    print("WARNING: SID threshold is provisional. Run training/evaluate_identification.py.")
