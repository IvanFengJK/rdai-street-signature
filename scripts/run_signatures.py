"""City visual signatures — the final analysis. No training.

  1. characteristic / atypical panels per city, by discriminative margin
  2. ~3 model-discovered visual modes per city, with exemplars
  3. per-mode interpretability + capture-confound summary

  python scripts/run_signatures.py [--cities Singapore Berlin ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
from PIL import Image                    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config                              # noqa: E402
from src.embed import load                                    # noqa: E402
from src.signatures import (city_modes, confound_flags,       # noqa: E402
                            discriminative_scores, summarise_modes)

OUT = Path("outputs/signatures")
N_PANEL = 6
N_MODES = 3


def strip(paths, titles, suptitle, dest, color="black"):
    n = len(paths)
    fig, ax = plt.subplots(1, n, figsize=(2.3 * n, 3.0))
    ax = np.atleast_1d(ax)
    for a, p, t in zip(ax, paths, titles):
        a.imshow(Image.open(p).convert("RGB"))
        a.set_title(t, fontsize=7, color=color)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    df = pd.read_parquet("data/interim/dataset.parquet").reset_index(drop=True)
    uu, emb = load(Path("data/interim/emb_trained.npz"))
    assert list(uu) == list(df.uuid), "embedding order mismatch"
    cities = sorted(df.city_ascii.unique())

    print("scoring discriminative margin over all 100,000 images ...")
    df["margin"] = discriminative_scores(emb, df, cities)
    print(f"  margin range {df.margin.min():.3f} .. {df.margin.max():.3f}\n")

    targets = args.cities or cities
    all_modes, all_rank = [], []

    for city in targets:
        m = (df.city_ascii == city).to_numpy()
        g = df[m].copy()

        # ---- 1. characteristic / atypical -----------------------------
        top = g.nlargest(N_PANEL, "margin")
        bot = g.nsmallest(N_PANEL, "margin")
        safe = city.replace(" ", "_")
        strip(top.path.tolist(), [f"margin {v:+.3f}" for v in top.margin],
              f"{city} — MOST characteristic (highest own-city margin)",
              OUT / f"{safe}_characteristic.png")
        strip(bot.path.tolist(), [f"margin {v:+.3f}" for v in bot.margin],
              f"{city} — ATYPICAL (lowest own-city margin)",
              OUT / f"{safe}_atypical.png", color="firebrick")
        for lbl, sel in [("characteristic", top), ("atypical", bot)]:
            for _, r in sel.iterrows():
                all_rank.append({"city": city, "kind": lbl, "uuid": r.uuid,
                                 "margin": r.margin, "year": r.year,
                                 "green_view_index": r.green_view_index,
                                 "building_view_index": r.building_view_index,
                                 "type_highway": r.type_highway})

        # ---- 2. model-discovered visual modes -------------------------
        labels, sim = city_modes(emb, m, N_MODES, seed)
        g = g.reset_index(drop=True)
        g["mode"], g["mode_sim"] = labels, sim
        for mode in range(N_MODES):
            ex = g[g["mode"] == mode].nlargest(N_PANEL, "mode_sim")
            if ex.empty:
                continue
            strip(ex.path.tolist(),
                  [f"gvi {v:.2f}" for v in ex.green_view_index],
                  f"{city} — model-discovered visual mode {mode} "
                  f"({(labels == mode).mean():.0%} of city)",
                  OUT / f"{safe}_mode{mode}.png")

        # ---- 3. interpretability + confound check ---------------------
        s = summarise_modes(g, labels)
        s["confound_flag"] = confound_flags(s, g)
        s.insert(0, "city", city)
        all_modes.append(s.reset_index())
        print(f"=== {city} ===")
        show = ["mode", "n", "share", "green_view_index", "building_view_index",
                "top_road", "top_year", "top_year_share",
                "top_projection_type", "top_projection_type_share",
                "confound_flag"]
        show = [c for c in show if c in s.reset_index().columns]
        print(s.reset_index()[show].round(3).to_string(index=False))
        print()

    modes = pd.concat(all_modes, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    modes.to_csv(OUT / "mode_summary.csv", index=False)
    pd.DataFrame(all_rank).to_csv(OUT / "characteristic_atypical.csv", index=False)

    flagged = modes[modes.confound_flag != "-"]
    print("=" * 66)
    print(f"CONFOUND CHECK: {len(flagged)} of {len(modes)} modes are concentrated "
          f"on a capture variable well beyond their city's own baseline")
    if len(flagged):
        print(flagged[["city", "mode", "share", "confound_flag"]]
              .round(3).to_string(index=False))
    print(f"\nwrote {OUT}/ ({len(list(OUT.glob('*.png')))} panels), "
          f"mode_summary.csv, characteristic_atypical.csv")


if __name__ == "__main__":
    main()
