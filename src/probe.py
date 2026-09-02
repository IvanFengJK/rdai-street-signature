"""Stage 11 — city linear probe.

A DIAGNOSTIC, never a training objective. Freeze an encoder, train one linear
layer to predict city from its embeddings, report validation accuracy against
5% chance.

The point is to measure how much city identity is linearly recoverable from
each representation, so that retrieval quality can be plotted against it. We do
NOT assume lower city accuracy is better — that is the relationship the
experiment is meant to measure, not assume.

Uses the same sequence-disjoint split as everything else, so the probe cannot
score itself on near-duplicate images of its own training streets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .dataset import build_split


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def city_probe(emb: np.ndarray, uuids, df: pd.DataFrame, classes: list[str],
               cfg: dict, device: str = "cuda", verbose: bool = False) -> dict:
    """Train a linear city classifier on frozen embeddings.

    `emb` rows must be aligned with `uuids`, which must align with `df`.
    """
    p = cfg["stage11"]["probe"]
    seed = cfg["project"]["seed"]

    assert len(emb) == len(df), "embedding/dataframe length mismatch"
    assert list(uuids) == list(df.uuid), "embedding order does not match dataframe"

    train_df, val_df = build_split(df, cfg["dataloader"]["val_fraction"], seed)
    overlap = set(train_df.sequence_id) & set(val_df.sequence_id)
    assert not overlap, f"RULE 1 VIOLATED in probe split: {len(overlap)} shared"

    pos = {u: i for i, u in enumerate(df.uuid)}
    tr = np.fromiter((pos[u] for u in train_df.uuid), int, len(train_df))
    va = np.fromiter((pos[u] for u in val_df.uuid), int, len(val_df))
    c2i = {c: i for i, c in enumerate(classes)}

    # Standardise then L2-normalise: puts every representation on the same
    # footing so probe accuracy reflects linear separability, not feature scale.
    X = emb.astype(np.float32)
    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    X = _l2((X - mu) / sd)

    Xtr = torch.from_numpy(X[tr]).to(device)
    ytr = torch.tensor([c2i[c] for c in train_df.city_ascii], device=device)
    Xva = torch.from_numpy(X[va]).to(device)
    yva = torch.tensor([c2i[c] for c in val_df.city_ascii], device=device)

    g = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)
    lin = nn.Linear(X.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=p["lr"],
                            weight_decay=p["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["epochs"])
    crit = nn.CrossEntropyLoss()
    bs = p["batch_size"]

    best = 0.0
    for ep in range(p["epochs"]):
        lin.train()
        perm = torch.randperm(len(Xtr), generator=g).to(device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            crit(lin(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
        lin.eval()
        with torch.no_grad():
            acc = (lin(Xva).argmax(1) == yva).float().mean().item()
        best = max(best, acc)
        if verbose and (ep + 1) % 10 == 0:
            print(f"    probe epoch {ep+1}/{p['epochs']}  val_acc={acc:.4f}")

    return {"city_probe_acc": acc, "city_probe_best": best,
            "chance": 1.0 / len(classes), "n_train": len(tr), "n_val": len(va)}
