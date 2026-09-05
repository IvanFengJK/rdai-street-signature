"""City visual signatures from the frozen Stage 5 embedding.

Answers "what makes an ordinary street look like this city, according to the
trained model?" — with the emphasis on *according to the model*. Nothing here
establishes that a pattern is an intrinsic property of the city; the confound
summary exists precisely so a mode that is really one camera or one month can
be recognised as such.

No training. Operates entirely on cached embeddings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Physical attributes (interpretable) vs capture attributes (confound check).
PHYSICAL = ["green_view_index", "building_view_index", "sky_view_index"]
CAPTURE = ["year", "month", "hour"]
CAPTURE_CAT = ["source", "projection_type"]
ROAD = "type_highway"


def l2norm(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)


def discriminative_scores(emb: np.ndarray, df: pd.DataFrame,
                          cities: list[str]) -> np.ndarray:
    """Margin score: similarity to own city minus best rival city.

        score(x) = cos(x, mu_own) - max_{c != own} cos(x, mu_c)

    Deliberately a MARGIN, not distance to the city centroid. Distance alone
    rewards images that are merely typical of the dataset; the margin rewards
    images that are typical of this city *and unlike every other city*, which
    is what "characteristic" should mean. High = characteristic, low (or
    negative) = atypical for its own city.
    """
    Z = l2norm(emb.astype(np.float32))
    idx = {c: i for i, c in enumerate(cities)}
    mus = l2norm(np.stack([Z[(df.city_ascii == c).to_numpy()].mean(0)
                           for c in cities]))
    sims = Z @ mus.T                                    # (N, n_cities)
    own = df.city_ascii.map(idx).to_numpy()
    own_sim = sims[np.arange(len(Z)), own]
    rival = sims.copy()
    rival[np.arange(len(Z)), own] = -np.inf
    return own_sim - rival.max(axis=1)


def city_modes(emb: np.ndarray, mask: np.ndarray, n_modes: int, seed: int):
    """K-means within one city. These are MODEL-DISCOVERED VISUAL MODES, not
    objective urban archetypes — they are whatever structure this particular
    embedding happens to carry."""
    from sklearn.cluster import KMeans
    Z = l2norm(emb[mask].astype(np.float32))
    km = KMeans(n_clusters=n_modes, random_state=seed, n_init=10).fit(Z)
    # rank members of each mode by closeness to its centroid, for exemplars
    cen = l2norm(km.cluster_centers_)
    sim = (Z * cen[km.labels_]).sum(1)
    return km.labels_, sim


def summarise_modes(df_city: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per-mode summary: interpretable physical attributes AND capture
    variables side by side, so a suspicious alignment is visible."""
    d = df_city.copy()
    d["mode"] = labels
    rows = []
    for m, g in d.groupby("mode"):
        r = {"mode": int(m), "n": len(g), "share": len(g) / len(d)}
        for a in PHYSICAL:
            if a in g:
                r[a] = float(g[a].mean())
        if ROAD in g:
            vc = g[ROAD].fillna("unknown").value_counts(normalize=True)
            r["top_road"] = vc.index[0]
            r["top_road_share"] = float(vc.iloc[0])
        for a in CAPTURE:
            if a in g:
                vc = g[a].value_counts(normalize=True)
                r[f"top_{a}"] = vc.index[0]
                r[f"top_{a}_share"] = float(vc.iloc[0])
        for a in CAPTURE_CAT:
            if a in g:
                vc = g[a].astype(str).value_counts(normalize=True)
                r[f"top_{a}"] = vc.index[0]
                r[f"top_{a}_share"] = float(vc.iloc[0])
        rows.append(r)
    return pd.DataFrame(rows).set_index("mode")


def confound_flags(summary: pd.DataFrame, df_city: pd.DataFrame,
                   abs_threshold: float = 0.60,
                   excess_threshold: float = 0.20) -> pd.Series:
    """Flag modes that look like a capture artefact rather than urban form.

    Measured as EXCESS concentration over the city's own baseline, not raw
    concentration. A city that is 100% perspective will have every mode at 100%
    perspective — that is a property of the city's collection, not evidence
    that this mode is a camera artefact. What matters is a mode being far more
    concentrated than the city as a whole: e.g. a mode that is 95% one year in
    a city whose top year is only 40%.

    Flagged when the mode's dominant category is both high in absolute terms
    and at least `excess_threshold` above the city-wide share of that same
    category.
    """
    flags = []
    for _, row in summary.iterrows():
        hits = []
        for var in [*CAPTURE, *CAPTURE_CAT]:
            share_col, top_col = f"top_{var}_share", f"top_{var}"
            if share_col not in summary.columns or var not in df_city:
                continue
            share = row.get(share_col, 0.0)
            if share < abs_threshold:
                continue
            base = float((df_city[var].astype(str) ==
                          str(row[top_col])).mean())      # city-wide baseline
            if share - base >= excess_threshold:
                hits.append(f"{var}(+{share - base:.2f})")
        flags.append(", ".join(hits) if hits else "-")
    return pd.Series(flags, index=summary.index, name="confound_flag")
