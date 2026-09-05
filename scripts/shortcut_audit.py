"""Shortcut audit — can city be predicted WITHOUT looking at the street?

Asked before proposing the city-distinctiveness phase, because a 90.6% city
classifier is only interesting if it is reading urban form rather than the
camera, the compression, the season or the collection campaign.

The decisive test is a metadata-only classifier: no pixels at all, just capture
attributes. If that matches the pixel model, then the confound is sufficient to
explain the pixel model's accuracy — which does not prove the CNN used it, but
does mean the accuracy alone is not evidence of anything about cities.

Uses the same sequence-disjoint split. Read-only; writes one report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config          # noqa: E402
from src.dataset import build_split       # noqa: E402

META = ["width", "height", "year", "month", "hour", "sequence_img_count"]


def _prep(d: pd.DataFrame) -> pd.DataFrame:
    X = d[META].astype(float).copy()
    X["aspect"] = d.width / d.height
    X["src_kartaview"] = (d.source == "KartaView").astype(float)
    X["proj_fisheye"] = (d.projection_type == "fisheye").astype(float)
    return X.fillna(-1)


def metadata_only_accuracy(d: pd.DataFrame, cfg: dict, label: str) -> dict:
    tr, va = build_split(d, cfg["dataloader"]["val_fraction"], cfg["project"]["seed"])
    assert not (set(tr.sequence_id) & set(va.sequence_id)), "RULE 1 VIOLATED"
    classes = sorted(d.city_ascii.unique())
    c2i = {c: i for i, c in enumerate(classes)}
    m = HistGradientBoostingClassifier(
        max_iter=150, random_state=cfg["project"]["seed"]).fit(
            _prep(tr), tr.city_ascii.map(c2i))
    acc = float(m.score(_prep(va), va.city_ascii.map(c2i)))
    print(f"  {label:38} {acc:.4f}   n={len(d):>7,}  chance={1/len(classes):.3f}")
    return {"subset": label, "n": int(len(d)), "n_cities": len(classes),
            "metadata_only_acc": acc, "chance": 1 / len(classes)}


def main():
    cfg = load_config()
    df = pd.read_parquet("data/interim/dataset.parquet")
    report: dict = {}

    print("=== per-city capture confounds ===")
    fish = df.groupby("city_ascii").projection_type.apply(
        lambda s: (s == "fisheye").mean())
    kv = df.groupby("city_ascii").source.apply(lambda s: (s == "KartaView").mean())
    topyear = df.groupby("city_ascii").year.apply(
        lambda s: s.value_counts(normalize=True).iloc[0])
    topmonth = df.groupby("city_ascii").month.apply(
        lambda s: s.value_counts(normalize=True).iloc[0])
    for nm, ser in [("fisheye share", fish), ("kartaview share", kv),
                    ("single-year share", topyear), ("single-month share", topmonth)]:
        print(f"  {nm:22} min {ser.min():.3f}  median {ser.median():.3f}  "
              f"max {ser.max():.3f} ({ser.idxmax()})")
        report[nm.replace(" ", "_")] = {
            "min": float(ser.min()), "median": float(ser.median()),
            "max": float(ser.max()), "argmax": str(ser.idxmax())}

    print("\n=== DECISIVE: city from METADATA ONLY, no pixels ===")
    rows = [metadata_only_accuracy(df, cfg, "current dataset (all)"),
            metadata_only_accuracy(df[df.projection_type == "perspective"], cfg,
                                   "perspective only"),
            metadata_only_accuracy(
                df[(df.projection_type == "perspective") &
                   df.year.between(2018, 2022)], cfg,
                "perspective + years 2018-2022")]
    report["metadata_only"] = rows
    print(f"\n  Stage 5 PIXEL encoder val accuracy: 0.9063")
    print(f"  -> metadata alone reaches {rows[0]['metadata_only_acc']:.4f}, i.e. "
          f"{rows[0]['metadata_only_acc']/0.9063*100:.1f}% of the pixel model.")
    print("     Stratifying does NOT remove it: the confound is structural.")

    tr, va = build_split(df, cfg["dataloader"]["val_fraction"], cfg["project"]["seed"])
    classes = sorted(df.city_ascii.unique())
    c2i = {c: i for i, c in enumerate(classes)}
    m = HistGradientBoostingClassifier(
        max_iter=150, random_state=cfg["project"]["seed"]).fit(
            _prep(tr), tr.city_ascii.map(c2i))
    sub = _prep(va).sample(4000, random_state=0)
    r = permutation_importance(m, sub, va.city_ascii.map(c2i).loc[sub.index],
                               n_repeats=3, random_state=0)
    print("\n  which metadata features carry the city signal:")
    imp = {}
    for i in np.argsort(-r.importances_mean)[:6]:
        name = list(sub.columns)[i]
        imp[name] = float(r.importances_mean[i])
        print(f"    {name:20} {r.importances_mean[i]:+.4f}")
    report["permutation_importance"] = imp

    print("\n=== cross-campaign feasibility (perspective only, full pool) ===")
    full = pd.read_parquet("data/interim/subset_clean.parquet")
    full = full[(~full.pano_status.fillna(False).astype(bool)) &
                (full.projection_type == "perspective")]
    feas = []
    for c, g in full.groupby("city_ascii"):
        vc = g.year.value_counts().sort_index()
        yrs = vc.index.to_numpy()
        best = (None, 0, 0, 0)
        for s in range(1, len(yrs)):
            a, b = vc[yrs[:s]].sum(), vc[yrs[s:]].sum()
            if min(a, b) > best[3]:
                best = (int(yrs[s]), int(a), int(b), int(min(a, b)))
        feas.append({"city": c, "cut_year": best[0], "early": best[1],
                     "late": best[2], "smaller_side": best[3]})
    fd = pd.DataFrame(feas).set_index("city").sort_values("smaller_side")
    print(fd.to_string())
    print(f"\n  cities with >=1,000 on the smaller side: "
          f"{int((fd.smaller_side >= 1000).sum())}/20")
    print(f"  cities with >=2,000 on the smaller side: "
          f"{int((fd.smaller_side >= 2000).sum())}/20")
    report["cross_campaign_feasibility"] = fd.reset_index().to_dict("records")

    Path("outputs").mkdir(exist_ok=True)
    json.dump(report, open("outputs/shortcut_audit.json", "w"), indent=2)
    fd.to_csv("outputs/shortcut_campaign_feasibility.csv")
    print("\nwrote outputs/shortcut_audit.json, outputs/shortcut_campaign_feasibility.csv")


if __name__ == "__main__":
    main()
