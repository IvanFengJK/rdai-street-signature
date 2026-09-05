"""Stage 7b — corrected cross-city retrieval evaluation.

Closes out the original research question honestly. The Stage 7 numbers in
outputs/stage7_results.csv are left untouched; this writes a separate,
clearly-labelled supplementary result.

Same 20 frozen queries, same physical attributes, same four representations.
The only change is the candidate search: exclude the query's own city first,
then take the true nearest 10 from everything that remains.

Also reports the original nearest-400 behaviour as a city-clustering diagnostic,
and a cross-city availability curve over candidate depth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config                                    # noqa: E402
from src.embed import load                                          # noqa: E402
from src.retrieval import (CATEGORICAL_ATTRS, NUMERIC_ATTRS,        # noqa: E402
                           cross_city_neighbours, random_control,
                           tabular_matrix)
from src.retrieval_corrected import (bootstrap_ci,                  # noqa: E402
                                     cross_city_neighbours_full,
                                     depth_diagnostics, paired_test,
                                     per_query_gaps, per_query_mismatch,
                                     slot_diagnostics)

N_BOOT = 10000
OUT = Path("outputs")


def main():
    cfg = load_config()
    seed = cfg["project"]["seed"]
    k = cfg["retrieval"]["top_k"]
    df = pd.read_parquet("data/interim/dataset.parquet").reset_index(drop=True)

    d = np.load("data/interim/stage7_queries.npz", allow_pickle=True)
    idx, q_idx = d["sub_idx"], d["q_idx"]
    sub = df.loc[idx].reset_index(drop=True)
    assert list(sub.loc[q_idx, "uuid"]) == list(d["query_uuid"]), \
        "frozen queries no longer point at the same images"
    print(f"{len(sub):,} candidates, {len(q_idx)} frozen queries, k={k}\n")

    _, E_city = load(Path("data/interim/emb_trained.npz"))
    _, E_inet = load(Path("data/interim/emb_imagenet.npz"))
    _, E_vic = load(Path("data/interim/emb_s11_vicreg_a_ep100.npz"))
    systems = {
        "Stage 5 city encoder": E_city[idx],
        "frozen imagenet": E_inet[idx],
        "VICReg ep100": E_vic[idx],
        "tabular (ORACLE-ish sanity check)": tabular_matrix(sub),
    }

    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(q_idx), size=(N_BOOT, len(q_idx)))

    # ---------------------------------------------------------------- 7b main
    rows, pq = [], {}
    for name, X in systems.items():
        nb = cross_city_neighbours_full(X, sub, q_idx, k)
        pq[name] = {a: per_query_gaps(sub, q_idx, nb, a) for a in NUMERIC_ATTRS}
        for a in CATEGORICAL_ATTRS:
            pq[name][f"{a}_mismatch"] = per_query_mismatch(sub, q_idx, nb, a)
        row = {"system": name}
        for metric, vals in pq[name].items():
            m, lo, hi = bootstrap_ci(vals, boot)
            row[metric] = m
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        row.update({f"slots_{kk}": vv
                    for kk, vv in slot_diagnostics(nb, k).items()})
        rows.append(row)

    ctl = random_control(sub, q_idx, k, seed)
    rows.append({"system": "random control", **ctl})
    table = pd.DataFrame(rows).set_index("system")

    metrics = [*NUMERIC_ATTRS, f"{CATEGORICAL_ATTRS[0]}_mismatch"]
    print("=== STAGE 7b — CORRECTED (exclude home city, then true top-10) ===")
    print("lower is better; 95% CI from 10,000 query bootstraps\n")
    for m in metrics:
        print(f"  {m}")
        for name in table.index:
            v = table.loc[name, m]
            lo = table.loc[name].get(f"{m}_ci_lo", np.nan)
            hi = table.loc[name].get(f"{m}_ci_hi", np.nan)
            ci = f"  [{lo:.4f}, {hi:.4f}]" if np.isfinite(lo) else ""
            print(f"    {name:36} {v:.4f}{ci}")
        print()

    # ------------------------------------------------------- paired contrasts
    print("=== PAIRED BOOTSTRAP CONTRASTS (negative = first system better) ===")
    contrasts = [("VICReg ep100", "frozen imagenet"),
                 ("VICReg ep100", "Stage 5 city encoder"),
                 ("frozen imagenet", "Stage 5 city encoder")]
    ctext = []
    for a, b in contrasts:
        for m in metrics:
            r = paired_test(pq[a][m], pq[b][m], boot)
            verdict = "SIGNIFICANT" if r["significant"] else "not significant"
            line = (f"  {a} - {b}, {m}: {r['diff']:+.4f} "
                    f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  {verdict}")
            print(line)
            ctext.append({"a": a, "b": b, "metric": m, **r})
        print()

    # ------------------------------------------- diagnostic: original top-400
    print("=== DIAGNOSTIC: original nearest-400-then-filter (Stage 7 behaviour) ===")
    print("This is the defect, quantified. It is a city-clustering measure.\n")
    diag = []
    for name, X in systems.items():
        s = slot_diagnostics(cross_city_neighbours(X, sub, q_idx, k), k)
        diag.append({"system": name, **s})
        print(f"  {name:36} zero-neighbour queries "
              f"{s['queries_with_zero_neighbours']:>2}/{s['n_queries']}   "
              f"slots filled {s['mean_slots_filled']:.2f}/{k}   "
              f"pairs {s['total_pairs']}/{s['max_possible_pairs']}")

    # ------------------------------------------------ availability vs depth
    print("\n=== CROSS-CITY AVAILABILITY vs CANDIDATE DEPTH ===")
    print("fraction of queries with at least k=10 cross-city candidates\n")
    depth_rows = []
    for name, X in systems.items():
        dd = depth_diagnostics(X, sub, q_idx, k)
        dd.insert(0, "system", name)
        depth_rows.append(dd)
        frac = "  ".join(
            f"d={int(row.depth)}: {row.frac_queries_with_k_plus_cross_city:.2f}"
            for row in dd.itertuples())
        print(f"  {name:36} {frac}")
    depth_df = pd.concat(depth_rows, ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "stage7b_corrected_results.csv")
    pd.DataFrame(diag).to_csv(OUT / "stage7b_clustering_diagnostic.csv", index=False)
    depth_df.to_csv(OUT / "stage7b_depth_curve.csv", index=False)
    json.dump(ctext, open(OUT / "stage7b_contrasts.json", "w"), indent=2)
    print(f"\nwrote stage7b_corrected_results.csv, stage7b_clustering_diagnostic.csv, "
          f"stage7b_depth_curve.csv, stage7b_contrasts.json")
    print("outputs/stage7_results.csv (original Stage 7) is untouched.")


if __name__ == "__main__":
    main()
