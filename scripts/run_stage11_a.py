"""Stage 11 Experiment A — VICReg, no adversary, no bottom mask.

Trains a fresh ResNet50 on the SAME sequence-disjoint train split used by
Stages 3-10, with the city-classification objective replaced by VICReg.

Writes only to checkpoints/stage11/. Nothing from Stages 0-10 is touched.

  python scripts/run_stage11_a.py [tag] [--epochs N]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config                    # noqa: E402
from src.dataset import build_split                 # noqa: E402
from src.ssl_dataset import make_ssl_loader         # noqa: E402
from src.ssl_model import build_ssl_model, describe  # noqa: E402
from src.train import set_seed                      # noqa: E402
from src.train_ssl import train_ssl                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag", nargs="?", default="vicreg_a")
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--limit", type=int, default=None,
                help="smoke test only: cap the number of training images")
args = ap.parse_args()

cfg = load_config()
if args.epochs is not None:
    cfg["stage11"]["ssl"]["epochs"] = args.epochs

# Seed BEFORE the model is built - the Stage 5 lesson. Building 23.5M weights
# from an unseeded RNG makes "same seed" meaningless.
set_seed(cfg["project"]["seed"])

df = pd.read_parquet("data/interim/dataset.parquet")
classes = sorted(df.city_ascii.unique())
train_df, val_df = build_split(df, cfg["dataloader"]["val_fraction"],
                               cfg["project"]["seed"])

# Rule 1, re-asserted here rather than assumed from Stage 3.
overlap = set(train_df.sequence_id) & set(val_df.sequence_id)
assert not overlap, f"RULE 1 VIOLATED: {len(overlap)} shared sequence_ids"

# Sanity: the evaluation attributes and perception scores must never reach
# training. The SSL dataset reads only `path` and `city_ascii`.
assert cfg["stage11"]["train_on"] == "train_split"
assert not cfg["stage11"]["adversarial"]["enabled"], \
    "Experiment A must run with the adversary OFF"
assert not cfg["stage11"]["augmentation"]["bottom_mask"]["enabled"], \
    "Experiment A must run with bottom_mask OFF (that is Experiment C)"

if args.limit:
    train_df = train_df.head(args.limit)
    val_df = val_df.head(max(256, args.limit // 4))
    print(f"*** SMOKE TEST: {len(train_df)} train / {len(val_df)} val ***")

train_loader = make_ssl_loader(train_df, classes, cfg, shuffle=True)
val_loader = make_ssl_loader(val_df, classes, cfg, shuffle=False)

model = build_ssl_model(cfg)
print(describe(model))
print(f"\ntrain {len(train_df):,} | val {len(val_df):,} | "
      f"batch {cfg['stage11']['ssl']['batch_size']} | "
      f"{len(train_loader)} steps/epoch | "
      f"{cfg['stage11']['ssl']['epochs']} epochs\n", flush=True)

train_ssl(model, train_loader, val_loader, cfg, tag=args.tag)
print("\nExperiment A training complete.")
