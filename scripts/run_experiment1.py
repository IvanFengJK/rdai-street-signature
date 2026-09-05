"""Experiment 1 — does the Stage 5 city signal survive a change of campaign?

A decision experiment, not an exploration. No training.

The shortcut audit showed city is 92.3% predictable from capture metadata alone
(year and month dominating), so the Stage 5 encoder's 90.6% is not by itself
evidence it learned anything about urban form. The clean test: train a linear
city probe on one capture period and test it on a DISJOINT later period of the
same cities. A representation that encodes the campaign cannot transfer; one
that encodes the city can.

Three conditions, one frozen encoder, one linear probe:
  A  original sequence-disjoint split          (the published 0.9063)
  B  cross-campaign: train early, test late
  C  metadata only, same cross-campaign split  (the confound's own score)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config          # noqa: E402
from src.dataset import build_split       # noqa: E402
from src.embed import load                # noqa: E402

META = ["width", "height", "year", "month", "hour", "sequence_img_count"]


def linear_probe(emb, tr_i, te_i, y_tr, y_te, n_classes, cfg, device="cuda"):
    """Same probe as src.probe, but with an explicitly supplied split."""
    p = cfg["stage11"]["probe"]
    seed = cfg["project"]["seed"]
    X = emb.astype(np.float32)
    mu, sd = X[tr_i].mean(0, keepdims=True), X[tr_i].std(0, keepdims=True) + 1e-6
    X = (X - mu) / sd
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)

    Xtr = torch.from_numpy(X[tr_i]).to(device)
    Xte = torch.from_numpy(X[te_i]).to(device)
    ytr = torch.tensor(y_tr, device=device)
    yte = torch.tensor(y_te, device=device)

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    lin = nn.Linear(X.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=p["lr"],
                            weight_decay=p["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["epochs"])
    crit = nn.CrossEntropyLoss()
    for _ in range(p["epochs"]):
        lin.train()
        perm = torch.randperm(len(Xtr), generator=g).to(device)
        for i in range(0, len(perm), p["batch_size"]):
            idx = perm[i:i + p["batch_size"]]
            opt.zero_grad(set_to_none=True)
            crit(lin(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
    lin.eval()
    with torch.no_grad():
        pred = lin(Xte).argmax(1)
    acc = (pred == yte).float().mean().item()
    per_city = {}
    for c in range(n_classes):
        m = yte == c
        if m.any():
            per_city[c] = (pred[m] == c).float().mean().item()
    return acc, per_city


def campaign_split(df: pd.DataFrame):
    """Per city, cut the year axis where it balances best. Early -> train."""
    early, late, cuts = [], [], {}
    for c, g in df.groupby("city_ascii"):
        vc = g.year.value_counts().sort_index()
        yrs = vc.index.to_numpy()
        best = (None, 0)
        for s in range(1, len(yrs)):
            sz = min(vc[yrs[:s]].sum(), vc[yrs[s:]].sum())
            if sz > best[1]:
                best = (yrs[s], sz)
        cut = best[0]
        if cut is None:                       # single-year city: cannot split
            cuts[c] = None
            continue
        cuts[c] = int(cut)
        early.append(g[g.year < cut])
        late.append(g[g.year >= cut])
    return pd.concat(early), pd.concat(late), cuts


def main():
    cfg = load_config()
    df = pd.read_parquet("data/interim/dataset.parquet").reset_index(drop=True)
    classes = sorted(df.city_ascii.unique())
    c2i = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    uu, emb = load(Path("data/interim/emb_trained.npz"))
    assert list(uu) == list(df.uuid), "embedding order mismatch"
    pos = {u: i for i, u in enumerate(df.uuid)}

    # ---- A: original sequence-disjoint split ----------------------------
    tr, va = build_split(df, cfg["dataloader"]["val_fraction"], cfg["project"]["seed"])
    assert not (set(tr.sequence_id) & set(va.sequence_id)), "RULE 1 VIOLATED"
    a_tr = np.array([pos[u] for u in tr.uuid])
    a_te = np.array([pos[u] for u in va.uuid])
    accA, _ = linear_probe(emb, a_tr, a_te,
                           tr.city_ascii.map(c2i).to_numpy(),
                           va.city_ascii.map(c2i).to_numpy(), n, cfg)

    # ---- B: cross-campaign ----------------------------------------------
    early, late, cuts = campaign_split(df)
    overlap = set(early.sequence_id) & set(late.sequence_id)
    print(f"campaign split: early {len(early):,} / late {len(late):,}")
    print(f"sequence overlap between campaigns: {len(overlap)}"
          f"  {'(OK)' if not overlap else '(!! not clean)'}")
    if overlap:                       # keep the non-negotiable guarantee
        late = late[~late.sequence_id.isin(overlap)]
        early = early[~early.sequence_id.isin(overlap)]
        print(f"  dropped overlapping sequences -> early {len(early):,} / late {len(late):,}")
    assert not (set(early.sequence_id) & set(late.sequence_id)), "RULE 1 VIOLATED"

    b_tr = np.array([pos[u] for u in early.uuid])
    b_te = np.array([pos[u] for u in late.uuid])
    accB, per_city = linear_probe(emb, b_tr, b_te,
                                  early.city_ascii.map(c2i).to_numpy(),
                                  late.city_ascii.map(c2i).to_numpy(), n, cfg)

    # ---- C: metadata only, same cross-campaign split ---------------------
    def prep(d):
        X = d[META].astype(float).copy()
        X["aspect"] = d.width / d.height
        X["src"] = (d.source == "KartaView").astype(float)
        X["proj"] = (d.projection_type == "fisheye").astype(float)
        return X.fillna(-1)
    mm = HistGradientBoostingClassifier(
        max_iter=150, random_state=cfg["project"]["seed"]).fit(
            prep(early), early.city_ascii.map(c2i))
    accC = float(mm.score(prep(late), late.city_ascii.map(c2i)))

    print("\n" + "=" * 62)
    print("EXPERIMENT 1 — Stage 5 encoder under a campaign change")
    print("=" * 62)
    print(f"  1. Stage 5, original sequence-disjoint split   {accA:.4f}")
    print(f"  2. Stage 5, cross-campaign (early -> late)     {accB:.4f}")
    print(f"  3. Metadata only, same cross-campaign split    {accC:.4f}")
    print(f"  4. Chance                                      {1/n:.4f}")
    print(f"\n  retained fraction of original accuracy: {accB/accA:.1%}")
    print(f"  margin over the metadata confound:      {accB-accC:+.4f}")

    pc = (pd.Series({classes[k]: v for k, v in per_city.items()})
          .sort_values().round(3))
    print("\n  per-city cross-campaign accuracy (worst 6 / best 3):")
    for c, v in list(pc.items())[:6]:
        print(f"    {c:16} {v:.3f}   (cut year {cuts.get(c)})")
    print("    ...")
    for c, v in list(pc.items())[-3:]:
        print(f"    {c:16} {v:.3f}   (cut year {cuts.get(c)})")

    json.dump({"A_sequence_disjoint": accA, "B_cross_campaign": accB,
               "C_metadata_only_cross_campaign": accC, "chance": 1 / n,
               "retained_fraction": accB / accA, "margin_over_metadata": accB - accC,
               "per_city": {classes[k]: v for k, v in per_city.items()},
               "cut_years": cuts, "n_early": int(len(early)), "n_late": int(len(late))},
              open("outputs/experiment1_cross_campaign.json", "w"), indent=2)
    print("\nwrote outputs/experiment1_cross_campaign.json")


if __name__ == "__main__":
    main()
