"""Stage 5 runner. Pass a tag; run twice with the same seed to test reproducibility."""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config
from src.dataset import build_split, make_loaders
from src.model import build_model, describe
from src.train import set_seed, train
from src.viz import plot_history, plot_confusion

tag = sys.argv[1] if len(sys.argv) > 1 else "run1"
cfg = load_config()

# Seed BEFORE anything touches an RNG. build_model() initialises 23.5M weights;
# if that happens before seeding, two runs from the "same seed" start from
# different weights and can never reproduce each other.
set_seed(cfg["project"]["seed"])

df = pd.read_parquet("data/interim/dataset.parquet")
classes = sorted(df.city_ascii.unique())
tr, va = build_split(df, cfg["dataloader"]["val_fraction"], cfg["project"]["seed"])
assert not (set(tr.sequence_id) & set(va.sequence_id)), "RULE 1 VIOLATED"
trl, val = make_loaders(tr, va, classes, cfg)
model = build_model(cfg, len(classes))
print(describe(model), flush=True)
print(f"\ntrain {len(tr):,} | val {len(va):,} | classes {len(classes)}\n", flush=True)
hist, cm, best = train(model, trl, val, cfg, classes, tag=tag)
print(f"\nBEST val_acc = {best:.4f}  (chance = {1/len(classes):.4f})")
out = Path(cfg["paths"]["outputs"])
plot_history(hist, out / f"{tag}_curves.png")
plot_confusion(cm, classes, out / f"{tag}_confusion.png")
print("wrote", out / f"{tag}_curves.png", "and", out / f"{tag}_confusion.png")
