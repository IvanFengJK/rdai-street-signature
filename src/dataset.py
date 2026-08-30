"""Stage 3 — dataset and dataloaders with a sequence-ID split.

CLAUDE.md rule 1: the train/val split is by sequence ID, never random.
Mapillary/KartaView images come in sequences shot metres apart, so a random
split puts near-duplicates on both sides and inflates accuracy meaninglessly.
`build_split` asserts the intersection is empty and raises if it is not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_path(images_root: Path, row) -> Path:
    return images_root / row.city_ascii.replace(" ", "_") / f"{row.uuid}.jpg"


def attach_paths(df: pd.DataFrame, images_root: Path) -> pd.DataFrame:
    """Keep only rows whose image actually landed on disk."""
    df = df.copy()
    df["path"] = [str(image_path(images_root, r)) for r in df.itertuples()]
    exists = np.fromiter((Path(p).exists() for p in df["path"]), bool, len(df))
    return df[exists].copy()


def build_split(df: pd.DataFrame, val_fraction: float, seed: int
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by sequence_id, stratified within each city.

    Stratifying per city matters because sequence density is very uneven
    (Moscow ~2 images/sequence in the sample, Casablanca ~86). A global
    sequence split would hand whole cities to one side.
    """
    rng = np.random.default_rng(seed)
    val_seqs: set = set()
    for _, g in df.groupby("city_ascii", sort=True):
        seqs = g.sequence_id.unique()
        rng.shuffle(seqs)
        counts = g.sequence_id.value_counts()
        target = val_fraction * len(g)
        taken = 0
        for s in seqs:
            if taken >= target:
                break
            val_seqs.add(s)
            taken += counts[s]

    val = df[df.sequence_id.isin(val_seqs)].copy()
    train = df[~df.sequence_id.isin(val_seqs)].copy()

    overlap = set(train.sequence_id) & set(val.sequence_id)
    assert not overlap, (
        f"RULE 1 VIOLATED: {len(overlap)} sequence_ids appear in both splits. "
        f"Examples: {list(overlap)[:5]}"
    )
    assert len(train) and len(val), "empty split"
    return train, val


class StreetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], train: bool,
                 image_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.train = train
        self.size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def _tensor(self, im: Image.Image) -> torch.Tensor:
        import torchvision.transforms.v2.functional as F
        if self.train:
            im = F.resize(im, self.size + 32)
            i, j, h, w = _rand_crop_params(im.size, self.size)
            im = im.crop((j, i, j + w, i + h))
            if np.random.rand() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            im = F.resize(im, self.size + 32)
            w, h = im.size
            left, top = (w - self.size) // 2, (h - self.size) // 2
            im = im.crop((left, top, left + self.size, top + self.size))
        x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())
        x = x.permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - mean) / std

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = Image.open(r["path"]).convert("RGB")
        return self._tensor(im), self.class_to_idx[r["city_ascii"]]


def _rand_crop_params(size, out):
    w, h = size
    i = np.random.randint(0, max(1, h - out + 1))
    j = np.random.randint(0, max(1, w - out + 1))
    return i, j, out, out


def make_loaders(train_df, val_df, classes, cfg):
    dl = cfg["dataloader"]
    common = dict(batch_size=dl["batch_size"], num_workers=dl["num_workers"],
                  pin_memory=True, persistent_workers=dl["num_workers"] > 0)
    tr = DataLoader(StreetDataset(train_df, classes, True, dl["image_size"]),
                    shuffle=True, drop_last=True, **common)
    va = DataLoader(StreetDataset(val_df, classes, False, dl["image_size"]),
                    shuffle=False, drop_last=False, **common)
    return tr, va
