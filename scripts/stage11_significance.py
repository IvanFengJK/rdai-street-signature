"""Bootstrap confidence intervals for the Stage 11 retrieval margins.

The metric aggregates 20 queries x 10 neighbours = 200 pairs. The VICReg vs
frozen-ImageNet margins are ~0.004-0.008, which is small enough that the point
estimates alone do not establish a difference. This resamples QUERIES (not
pairs — pairs within a query are not independent) to put an interval on each
system and on the paired difference.

Read-only: uses cached embeddings and the frozen query set, writes nothing
except its own report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config                                # noqa: E402
from src.embed import load                                      # noqa: E402
from src.retrieval import (NUMERIC_ATTRS, cross_city_neighbours,  # noqa: E402
                           tabular_matrix)

N_BOOT = 10000


def per_query_gaps(df, q_idx, nb, attr) -> np.ndarray:
    """Mean |attribute difference| for each query separately."""
    v = df[attr].to_numpy(float)
    out = []
    for qi, row in zip(q_idx, nb):
        d = [abs(v[qi] - v[n]) for n in row
             if n >= 0 and np.isfinite(v[qi]) and np.isfinite(v[n])]
        out.append(np.mean(d) if d else np.nan)
    return np.asarray(out, float)


def main():
    cfg = load_config()
    seed = cfg["project"]["seed"]
    k = cfg["retrieval"]["top_k"]
    df = pd.read_parquet("data/interim/dataset.parquet").reset_index(drop=True)

    d = np.load("data/interim/stage7_queries.npz", allow_pickle=True)
    idx, q_idx = d["sub_idx"], d["q_idx"]
    sub = df.loc[idx].reset_index(drop=True)

    _, E_city = load(Path("data/interim/emb_trained.npz"))
    _, E_inet = load(Path("data/interim/emb_imagenet.npz"))
    _, E_vic = load(Path("data/interim/emb_s11_vicreg_a_ep100.npz"))

    systems = {
        "Stage 5 city encoder": E_city[idx],
        "frozen imagenet": E_inet[idx],
        "VICReg ep100": E_vic[idx],
        "tabular (oracle-ish)": tabular_matrix(sub),
    }

    rng = np.random.default_rng(seed)
    gaps: dict[str, dict[str, np.ndarray]] = {}
    for name, X in systems.items():
        nb = cross_city_neighbours(X, sub, q_idx, k)
        gaps[name] = {a: per_query_gaps(sub, q_idx, nb, a) for a in NUMERIC_ATTRS}

    nq = len(q_idx)
    boot = rng.integers(0, nq, size=(N_BOOT, nq))

    print(f"bootstrap over {nq} queries, {N_BOOT:,} resamples\n")
    for attr in NUMERIC_ATTRS:
        print(f"=== {attr} (lower is better) ===")
        for name in systems:
            g = gaps[name][attr]
            bs = np.nanmean(g[boot], axis=1)
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            print(f"  {name:24} {np.nanmean(g):.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

        a = gaps["VICReg ep100"][attr]
        b = gaps["frozen imagenet"][attr]
        diff = a - b                       # negative = VICReg better
        bs = np.nanmean(diff[boot], axis=1)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        p = float(np.mean(bs >= 0))        # P(VICReg no better), one-sided
        print(f"  --> paired diff (VICReg - imagenet) = {np.nanmean(diff):+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(no better)={p:.3f}")
        print(f"      {'SIGNIFICANT at 95%' if hi < 0 else 'NOT significant - CI crosses zero'}\n")


if __name__ == "__main__":
    main()
