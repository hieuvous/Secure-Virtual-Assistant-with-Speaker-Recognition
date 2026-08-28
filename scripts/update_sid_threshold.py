import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument("--sid-json",required=True);a=p.parse_args()
sid=json.loads(Path(a.sid_json).read_text())
path=ROOT/"configs/thresholds.json";cfg=json.loads(path.read_text())
cfg["sid_threshold"]=float(sid["sid_threshold"])
cfg["sid_status"]="TUNED_FROM_DEV"
cfg["sid_source"]=str(Path(a.sid_json))
path.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
print(json.dumps(cfg,indent=2))
