"""Stage 7 — three retrieval systems and the cross-city attribute metric.

Systems (CLAUDE.md rule 4), all answering the same query set:
  1. trained    - our encoder's embedding
  2. tabular    - normalised tabular feature vector (the sklearn baseline, rule 3)
  3. imagenet   - frozen ImageNet features, no training

Metric (rule 5): for each query, take the top-k neighbours *in other cities*
and measure mean absolute difference in physical attributes, against randomly
paired streets as the control. A good embedding retrieves streets matching on
attributes it was never shown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Physical attributes only. Never the perception scores (rule 2).
NUMERIC_ATTRS = ["green_view_index", "building_view_index"]
CATEGORICAL_ATTRS = ["type_highway"]


def tabular_matrix(df: pd.DataFrame) -> np.ndarray:
    """Normalised tabular feature vector for the baseline system.

    Deliberately excludes city, lat/lon and sequence: those would let the
    baseline retrieve on geography rather than streetscape, which is not what
    is being compared.
    """
    num = ["green_view_index", "sky_view_index", "building_view_index",
           "Vegetation", "Sky", "Building", "Total", "urban_code",
           "snap_dist", "lanes" if "lanes" in df.columns else "urban_code"]
    num = [c for c in dict.fromkeys(num) if c in df.columns]
    X = df[num].astype(float).fillna(df[num].astype(float).median())
    cat = pd.get_dummies(df["type_highway"].fillna("unknown"), prefix="hw")
    M = np.hstack([StandardScaler().fit_transform(X), cat.to_numpy(float)])
    return M


def l2norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-8, None)


def cross_city_neighbours(X: np.ndarray, df: pd.DataFrame, q_idx: np.ndarray,
                          k: int) -> np.ndarray:
    """Top-k neighbours of each query that lie in a DIFFERENT city."""
    Xn = l2norm(X)
    cities = df["city_ascii"].to_numpy()
    nn = NearestNeighbors(n_neighbors=min(len(df), k * 40), metric="cosine").fit(Xn)
    _, idx = nn.kneighbors(Xn[q_idx])
    out = np.full((len(q_idx), k), -1, dtype=int)
    for r, (qi, cand) in enumerate(zip(q_idx, idx)):
        keep = [c for c in cand if cities[c] != cities[qi]][:k]
        out[r, :len(keep)] = keep
    return out


def attribute_gap(df: pd.DataFrame, q_idx: np.ndarray, nb: np.ndarray) -> dict:
    """Mean absolute difference in physical attributes, query vs neighbour."""
    res = {}
    for a in NUMERIC_ATTRS:
        v = df[a].to_numpy(float)
        diffs = [abs(v[qi] - v[n]) for qi, row in zip(q_idx, nb)
                 for n in row if n >= 0 and np.isfinite(v[qi]) and np.isfinite(v[n])]
        res[a] = float(np.mean(diffs)) if diffs else float("nan")
    for a in CATEGORICAL_ATTRS:
        v = df[a].fillna("unknown").to_numpy()
        m = [float(v[qi] != v[n]) for qi, row in zip(q_idx, nb) for n in row if n >= 0]
        res[f"{a}_mismatch"] = float(np.mean(m)) if m else float("nan")
    return res


def random_control(df: pd.DataFrame, q_idx: np.ndarray, k: int, seed: int) -> dict:
    """Control: same queries, randomly paired partners in other cities."""
    rng = np.random.default_rng(seed)
    cities = df["city_ascii"].to_numpy()
    nb = np.full((len(q_idx), k), -1, dtype=int)
    all_idx = np.arange(len(df))
    for r, qi in enumerate(q_idx):
        pool = all_idx[cities != cities[qi]]
        nb[r] = rng.choice(pool, size=k, replace=False)
    return attribute_gap(df, q_idx, nb)


def evaluate_systems(systems: dict[str, np.ndarray], df: pd.DataFrame,
                     q_idx: np.ndarray, k: int, seed: int) -> pd.DataFrame:
    rows = []
    for name, X in systems.items():
        nb = cross_city_neighbours(X, df, q_idx, k)
        r = attribute_gap(df, q_idx, nb)
        r["system"] = name
        rows.append(r)
    ctl = random_control(df, q_idx, k, seed)
    ctl["system"] = "random control"
    rows.append(ctl)
    out = pd.DataFrame(rows).set_index("system")
    return out
