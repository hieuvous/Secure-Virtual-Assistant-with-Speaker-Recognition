import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.speaker.embedding import extract_embedding
from src.speaker.model import get_ecapa

parser = argparse.ArgumentParser()
parser.add_argument("audio")
args = parser.parse_args()

emb = extract_embedding(args.audio)
service = get_ecapa()
print("ECAPA OK")
print("Using fine-tuned:", service.using_finetuned)
print("Embedding shape:", emb.shape)
print("L2 norm:", float((emb ** 2).sum() ** 0.5))
