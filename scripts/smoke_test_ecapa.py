import argparse, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.speaker.embedding import extract_embedding
from src.speaker.model import get_ecapa

p=argparse.ArgumentParser();p.add_argument("audio");a=p.parse_args()
e=extract_embedding(a.audio,use_vad=True);s=get_ecapa()
print("ECAPA + VAD OK")
print("Using fine-tuned:",s.using_finetuned)
print("Checkpoint:",s.checkpoint_path)
print("Metadata:",s.checkpoint_metadata)
print("Embedding shape:",e.shape)
print("L2 norm:",float((e**2).sum()**0.5))
