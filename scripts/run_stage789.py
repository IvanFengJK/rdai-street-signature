"""Stages 7, 8 and 9 — retrieval + evaluation, UMAP, Grad-CAM."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config                                  # noqa: E402
from src.embed import load                                        # noqa: E402
from src.model import build_model                                 # noqa: E402
from src.retrieval import (cross_city_neighbours, evaluate_systems,  # noqa: E402
                           tabular_matrix)
from src.viz import (gradcam_panel, plot_pair_cards, plot_pairs_map,  # noqa: E402
                     plot_umap, umap_project)

cfg = load_config()
seed = cfg["project"]["seed"]
out = Path(cfg["paths"]["outputs"])
out.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet("data/interim/dataset.parquet").reset_index(drop=True)
classes = sorted(df.city_ascii.unique())

u_tr, E_tr = load(Path("data/interim/emb_trained.npz"))
u_in, E_in = load(Path("data/interim/emb_imagenet.npz"))
assert list(u_tr) == list(df.uuid), "trained embedding order does not match manifest"
assert list(u_in) == list(df.uuid), "imagenet embedding order does not match manifest"

# ---------------------------------------------------------------- Stage 7
# Retrieval runs over a stratified subsample: cosine kNN over 100k x 2048 for
# every system is needlessly heavy, and the metric is a mean over queries.
rng = np.random.default_rng(seed)
idx = (df.groupby("city_ascii", group_keys=False)
         .apply(lambda g: g.sample(min(len(g), 1500), random_state=seed))
         .index.to_numpy())
idx = np.sort(idx)
sub = df.loc[idx].reset_index(drop=True)
systems = {
    "trained (ours)": E_tr[idx],
    "tabular (baseline)": tabular_matrix(sub),
    "frozen imagenet": E_in[idx],
}
q_idx = rng.choice(len(sub), size=cfg["retrieval"]["n_queries"], replace=False)
k = cfg["retrieval"]["top_k"]
table = evaluate_systems(systems, sub, q_idx, k, seed)
print("\n=== STAGE 7 — cross-city retrieval, mean |attribute difference| ===")
print("(lower is better for the two numeric rows; random control is the "
      "no-skill baseline)\n")
print(table.round(4).to_string())
table.round(5).to_csv(out / "stage7_results.csv")

nb = cross_city_neighbours(systems["trained (ours)"], sub, q_idx, k)
plot_pairs_map(sub, q_idx, nb, out / "stage7_map.png")
plot_pair_cards(sub, q_idx, nb, out / "stage7_pairs.png", n=6)
print("wrote stage7_map.png, stage7_pairs.png, stage7_results.csv")

# ---------------------------------------------------------------- Stage 8
vis = np.sort(rng.choice(len(df), size=min(25000, len(df)), replace=False))
xy = umap_project(E_tr[vis], seed)
plot_umap(xy, df.loc[vis].reset_index(drop=True), out / "stage8_umap.png")
print("wrote stage8_umap.png")

# ---------------------------------------------------------------- Stage 9
ck = torch.load(Path(cfg["paths"]["checkpoints"]) / "runA_best.pt",
                map_location="cpu", weights_only=False)
model = build_model(cfg, len(classes))
model.load_state_dict(ck["model"])
panel = df.groupby("city_ascii", group_keys=False).head(1).head(8).reset_index(drop=True)
gradcam_panel(model, panel, cfg, out / "stage9_gradcam.png", n=len(panel))
print("wrote stage9_gradcam.png")

json.dump({"stage7": json.loads(table.to_json(orient="index"))},
          open(out / "stage789_summary.json", "w"), indent=2)
print("\nall done")
