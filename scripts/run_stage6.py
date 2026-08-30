"""Stage 6 — strip the head, extract and cache embeddings for every image.

Caches three embedding sets, because Stage 7 compares three retrieval systems:
  trained   - our encoder (backbone output, head bypassed)
  imagenet  - the same architecture with frozen ImageNet weights, no training
  (the tabular vector is built in Stage 7 straight from the dataframe)
"""
import sys
import time
from pathlib import Path

import pandas as pd
import timm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config          # noqa: E402
from src.embed import extract, save       # noqa: E402
from src.model import CityEncoder, build_model  # noqa: E402

cfg = load_config()
df = pd.read_parquet("data/interim/dataset.parquet")
classes = sorted(df.city_ascii.unique())
out = Path("data/interim")

ckpt_path = Path(cfg["paths"]["checkpoints"]) / "runA_best.pt"
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"loaded {ckpt_path} (epoch {ck['epoch']}, val_acc {ck['val_acc']:.4f})")

# ---- ours -----------------------------------------------------------------
model = build_model(cfg, len(classes))
model.load_state_dict(ck["model"])
t0 = time.time()
emb = extract(model, df, classes, cfg)
print(f"trained embeddings {emb.shape} in {(time.time()-t0)/60:.1f} min")
save(out / "emb_trained.npz", df.uuid.to_numpy(), emb)

# ---- frozen ImageNet, no training -----------------------------------------
inet = CityEncoder(cfg["encoder"]["backbone"], len(classes), pretrained=True)
t0 = time.time()
emb_i = extract(inet, df, classes, cfg)
print(f"imagenet embeddings {emb_i.shape} in {(time.time()-t0)/60:.1f} min")
save(out / "emb_imagenet.npz", df.uuid.to_numpy(), emb_i)
print("done")
