"""Stage 6 — strip the head, extract and cache embeddings for every image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from .dataset import StreetDataset


@torch.no_grad()
def extract(model, df: pd.DataFrame, classes: list[str], cfg: dict,
            device: str = "cuda", batch_size: int | None = None) -> np.ndarray:
    """Embeddings from the backbone with the classification head bypassed."""
    model = model.to(device).eval()
    ds = StreetDataset(df, classes, train=False,
                       image_size=cfg["dataloader"]["image_size"])
    dl = DataLoader(ds, batch_size=batch_size or cfg["dataloader"]["batch_size"],
                    shuffle=False, num_workers=cfg["dataloader"]["num_workers"],
                    pin_memory=True)
    out = []
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.float16):
            z = model.forward_features(x)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, 0)


def save(path: Path, uuids, emb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, uuid=np.asarray(uuids, dtype=object), emb=emb)


def load(path: Path):
    d = np.load(path, allow_pickle=True)
    return d["uuid"], d["emb"]
