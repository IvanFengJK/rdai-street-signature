"""Stage 11 evaluation — the Stage 7 protocol, unchanged, on a new encoder.

Reuses src.retrieval verbatim. The 30,000-image subsample and the 20 queries
come from the frozen file written by scripts/freeze_stage7_queries.py and are
asserted, never re-derived, so they cannot drift between experiments.

Adds the city linear probe so retrieval quality and recoverable city
information appear in one table.

  python scripts/run_stage11_eval.py --ckpt checkpoints/stage11/vicreg_a_epoch030.pt \
                                     --tag vicreg_a_ep030

Results append to outputs/stage11_results.csv. Nothing from Stages 0-10 is
overwritten; embeddings are cached as data/interim/emb_s11_<tag>.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config                                   # noqa: E402
from src.embed import extract, load, save                          # noqa: E402
from src.model import CityEncoder                                  # noqa: E402
from src.probe import city_probe                                   # noqa: E402
from src.retrieval import evaluate_systems, tabular_matrix         # noqa: E402
from src.ssl_model import SSLEncoder                               # noqa: E402

RESULTS = Path("outputs/stage11_results.csv")
QUERIES = Path("data/interim/stage7_queries.npz")


def frozen_queries(cfg, df):
    """Load the frozen Stage 7 query set and assert it still matches."""
    assert QUERIES.exists(), (
        f"{QUERIES} missing - run scripts/freeze_stage7_queries.py first")
    d = np.load(QUERIES, allow_pickle=True)
    idx, q_idx = d["sub_idx"], d["q_idx"]
    sub = df.loc[idx].reset_index(drop=True)
    assert list(sub.loc[q_idx, "uuid"]) == list(d["query_uuid"]), \
        "FROZEN QUERIES NO LONGER POINT AT THE SAME IMAGES"
    assert len(q_idx) == cfg["retrieval"]["n_queries"]
    return idx, q_idx, sub


def ssl_embeddings(ckpt_path: Path, tag: str, cfg, df, classes) -> np.ndarray:
    """Extract (or reuse cached) embeddings for an SSL checkpoint."""
    cache = Path(cfg["stage11"]["emb_dir"]) / f"emb_s11_{tag}.npz"
    if cache.exists():
        u, e = load(cache)
        assert list(u) == list(df.uuid), f"{cache} order mismatch"
        print(f"reusing cached {cache}  {e.shape}")
        return e
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"loaded {ckpt_path} (epoch {ck['epoch']})")
    model = SSLEncoder(cfg["encoder"]["backbone"], cfg["stage11"]["projector"])
    model.load_state_dict(ck["model"])
    t0 = time.time()
    emb = extract(model, df, classes, cfg)          # head bypassed by design
    print(f"extracted {emb.shape} in {(time.time()-t0)/60:.1f} min")
    save(cache, df.uuid.to_numpy(), emb)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["project"]["seed"]
    df = pd.read_parquet("data/interim/dataset.parquet").reset_index(drop=True)
    classes = sorted(df.city_ascii.unique())
    idx, q_idx, sub = frozen_queries(cfg, df)

    # --- the new representation -------------------------------------------
    E_new = ssl_embeddings(Path(args.ckpt), args.tag, cfg, df, classes)

    # --- baselines, exactly as Stage 7 built them --------------------------
    u_tr, E_city = load(Path("data/interim/emb_trained.npz"))
    u_in, E_inet = load(Path("data/interim/emb_imagenet.npz"))
    assert list(u_tr) == list(df.uuid) and list(u_in) == list(df.uuid)

    systems = {
        "trained city encoder (Stage 5)": E_city[idx],
        "frozen imagenet": E_inet[idx],
        f"VICReg [{args.tag}]": E_new[idx],
        "tabular (ORACLE-ish sanity check)": tabular_matrix(sub),
    }
    table = evaluate_systems(systems, sub, q_idx, cfg["retrieval"]["top_k"], seed)

    # --- city linear probe -------------------------------------------------
    probes: dict[str, float] = {}
    if not args.skip_probe:
        for name, E in [("trained city encoder (Stage 5)", E_city),
                        ("frozen imagenet", E_inet),
                        (f"VICReg [{args.tag}]", E_new)]:
            r = city_probe(E, df.uuid.to_numpy(), df, classes, cfg)
            probes[name] = r["city_probe_acc"]
            print(f"  city probe {name:36} val_acc={r['city_probe_acc']:.4f}")

    table["city_probe_acc"] = [probes.get(s, np.nan) for s in table.index]
    table["checkpoint"] = args.tag

    print("\n=== STAGE 11 EVALUATION (Stage 7 protocol, unchanged) ===")
    print("lower is better for the three attribute columns; "
          "city_probe chance = 0.05\n")
    print(table.round(4).to_string())

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    out = table.reset_index()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS)
        prev = prev[prev.checkpoint != args.tag]          # idempotent re-run
        out = pd.concat([prev, out], ignore_index=True)
    out.to_csv(RESULTS, index=False)
    print(f"\nappended to {RESULTS}")


if __name__ == "__main__":
    main()
