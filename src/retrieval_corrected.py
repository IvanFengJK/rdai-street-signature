"""Stage 7b — corrected cross-city neighbour search.

`src.retrieval.cross_city_neighbours` (Stage 7) retrieves the k*40 = 400
globally nearest candidates and THEN filters to other cities. For a
representation that clusters cities tightly this is degenerate: all 400 nearest
neighbours lie in the query's own city, so the query yields nothing. Measured on
the Stage 5 city encoder, 12 of 20 queries returned zero cross-city neighbours
and the average query filled only 2.35 of its 10 slots — so its reported score
came from 8 queries and ~47 pairs, and those 8 were the biased subset that
happened to sit near a city boundary.

The correction: exclude the query's own city FIRST, then take the true nearest
k from the entire remaining candidate set.

src.retrieval is imported, never modified — Stage 7's committed numbers stay
reproducible. Only the candidate search is replaced; `attribute_gap` and
`random_control` were never affected by the defect and are reused as-is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .retrieval import l2norm


def cross_city_neighbours_full(X: np.ndarray, df: pd.DataFrame,
                               q_idx: np.ndarray, k: int) -> np.ndarray:
    """True top-k neighbours from all candidates outside the query's city."""
    Xn = l2norm(X.astype(np.float32))
    cities = df["city_ascii"].to_numpy()
    sims = Xn[q_idx] @ Xn.T
    out = np.full((len(q_idx), k), -1, dtype=int)
    for r, qi in enumerate(q_idx):
        s = sims[r].copy()
        s[cities == cities[qi]] = -np.inf      # exclude the whole home city
        s[qi] = -np.inf
        top = np.argpartition(-s, k)[:k]
        out[r] = top[np.argsort(-s[top])]
    return out


def depth_diagnostics(X: np.ndarray, df: pd.DataFrame, q_idx: np.ndarray,
                      k: int, depths=(50, 100, 200, 400, 1000)) -> pd.DataFrame:
    """Cross-city availability as a function of candidate depth.

    At each depth d: take the d globally nearest candidates and ask how many
    lie outside the query's city. This measures how tightly a representation
    clusters cities — the tighter the clustering, the deeper you must search
    before any cross-city neighbour appears at all.
    """
    Xn = l2norm(X.astype(np.float32))
    cities = df["city_ascii"].to_numpy()
    sims = Xn[q_idx] @ Xn.T
    order = np.argsort(-sims, axis=1)

    rows = []
    for d in depths:
        avail = []
        for r, qi in enumerate(q_idx):
            cand = order[r, :d]
            cand = cand[cand != qi]
            avail.append(int((cities[cand] != cities[qi]).sum()))
        avail = np.asarray(avail)
        rows.append({
            "depth": d,
            "frac_queries_with_k_plus_cross_city": float((avail >= k).mean()),
            "frac_queries_with_zero": float((avail == 0).mean()),
            "mean_cross_city_available": float(avail.mean()),
            "median_cross_city_available": float(np.median(avail)),
        })
    return pd.DataFrame(rows)


def slot_diagnostics(nb: np.ndarray, k: int) -> dict:
    """How many of the k slots a neighbour matrix actually filled."""
    filled = (nb >= 0).sum(axis=1)
    return {
        "queries_with_zero_neighbours": int((filled == 0).sum()),
        "n_queries": int(len(nb)),
        "mean_slots_filled": float(filled.mean()),
        "total_pairs": int(filled.sum()),
        "max_possible_pairs": int(len(nb) * k),
    }


def per_query_gaps(df: pd.DataFrame, q_idx: np.ndarray, nb: np.ndarray,
                   attr: str) -> np.ndarray:
    """Mean |attribute difference| per query — the unit the bootstrap resamples.

    Queries, not pairs: pairs within a query share a query image and are not
    independent.
    """
    v = df[attr].to_numpy(float)
    out = []
    for qi, row in zip(q_idx, nb):
        d = [abs(v[qi] - v[n]) for n in row
             if n >= 0 and np.isfinite(v[qi]) and np.isfinite(v[n])]
        out.append(np.mean(d) if d else np.nan)
    return np.asarray(out, dtype=float)


def per_query_mismatch(df: pd.DataFrame, q_idx: np.ndarray, nb: np.ndarray,
                       attr: str) -> np.ndarray:
    """Per-query categorical mismatch rate."""
    v = df[attr].fillna("unknown").to_numpy()
    out = []
    for qi, row in zip(q_idx, nb):
        m = [float(v[qi] != v[n]) for n in row if n >= 0]
        out.append(np.mean(m) if m else np.nan)
    return np.asarray(out, dtype=float)


def bootstrap_ci(per_query: np.ndarray, boot: np.ndarray) -> tuple[float, float, float]:
    """(mean, lo, hi) from a precomputed query-resampling index matrix."""
    bs = np.nanmean(per_query[boot], axis=1)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(np.nanmean(per_query)), float(lo), float(hi)


def paired_test(a: np.ndarray, b: np.ndarray, boot: np.ndarray) -> dict:
    """Paired bootstrap of (a - b). Negative means `a` is better (lower gap)."""
    diff = a - b
    bs = np.nanmean(diff[boot], axis=1)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {
        "diff": float(np.nanmean(diff)),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "p_a_not_better": float(np.mean(bs >= 0)),
        "significant": bool(hi < 0),
    }
