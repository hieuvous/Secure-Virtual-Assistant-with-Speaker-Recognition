import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.database.db import init_db
from src.database.repositories import get_profile
from src.speaker.verification import verify_speaker
p=argparse.ArgumentParser();p.add_argument("--user-id",type=int,required=True);p.add_argument("--audio",required=True)
a=p.parse_args();init_db();pr=get_profile(a.user_id)
if not pr:raise RuntimeError("User has no enrolled speaker profile.")
print(json.dumps(verify_speaker(a.audio,a.user_id,pr["embedding_path"]),indent=2))
