"""Freeze the Stage 7 evaluation query set to disk.

Stage 7 derived its 30,000-image stratified subsample and its 20 queries from
the seed at runtime rather than persisting them. That is reproducible, but it
is reproducible only as long as nobody touches the derivation. Stage 11 must
answer *exactly* the same queries, so this writes them out once and every later
script asserts against the frozen file instead of re-deriving.

Re-running this is idempotent: if the file exists, it re-derives and asserts
equality rather than overwriting.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config          # noqa: E402

OUT = Path("data/interim/stage7_queries.npz")


def derive(cfg, df):
    """Byte-for-byte the derivation in scripts/run_stage789.py lines 40-56."""
    seed = cfg["project"]["seed"]
    rng = np.random.default_rng(seed)
    idx = (df.groupby("city_ascii", group_keys=False)
             .apply(lambda g: g.sample(min(len(g), 1500), random_state=seed))
             .index.to_numpy())
    idx = np.sort(idx)
    sub = df.loc[idx].reset_index(drop=True)
    q_idx = rng.choice(len(sub), size=cfg["retrieval"]["n_queries"], replace=False)
    return idx, q_idx


def main():
    cfg = load_config()
    df = pd.read_parquet("data/interim/dataset.parquet").reset_index(drop=True)
    idx, q_idx = derive(cfg, df)

    if OUT.exists():
        d = np.load(OUT)
        assert np.array_equal(d["sub_idx"], idx), \
            "STAGE 7 SUBSAMPLE DRIFTED from the frozen query file"
        assert np.array_equal(d["q_idx"], q_idx), \
            "STAGE 7 QUERIES DRIFTED from the frozen query file"
        print(f"{OUT} already frozen and still matches "
              f"({len(idx):,} subsample, {len(q_idx)} queries)")
        return

    sub = df.loc[idx].reset_index(drop=True)
    np.savez(OUT, sub_idx=idx, q_idx=q_idx,
             query_uuid=sub.loc[q_idx, "uuid"].to_numpy(dtype=object),
             query_city=sub.loc[q_idx, "city_ascii"].to_numpy(dtype=object))
    print(f"froze {len(idx):,}-image subsample and {len(q_idx)} queries -> {OUT}")
    print("query cities:", list(sub.loc[q_idx, "city_ascii"]))


if __name__ == "__main__":
    main()
