"""Stage 11 — two-view dataset for self-supervised training.

Stage 3's `StreetDataset` returns a single view with a fixed augmentation, so
it cannot serve SSL. This is a separate class; `StreetDataset` is left exactly
as Stages 3-10 use it.

The city index is returned alongside the two views. Experiment A ignores it;
Experiment B feeds it to the adversary. Returning it from the start means the
dataset does not have to change between experiments.

Nothing here touches the evaluation attributes (greenery, building density,
road type) or the prohibited perception scores. Only pixels and, for the
adversary, the city label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from .dataset import IMAGENET_MEAN, IMAGENET_STD, _worker_init


class BottomMask(torch.nn.Module):
    """Zero the bottom fraction of a normalised image tensor.

    Stage 9 found the city classifier attending to bottom-of-frame camera
    furniture — dashcam overlay strips, vehicle bonnets. After normalisation,
    zero is the dataset mean, so this replaces that band with "average colour"
    rather than injecting a black edge the model could latch onto instead.

    Off for Experiment A. Enabling it is Experiment C.
    """

    def __init__(self, probability: float, min_fraction: float,
                 max_fraction: float):
        super().__init__()
        self.p = probability
        self.lo = min_fraction
        self.hi = max_fraction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) >= self.p:
            return x
        h = x.shape[-2]
        frac = torch.empty(()).uniform_(self.lo, self.hi).item()
        rows = max(1, int(round(h * frac)))
        x = x.clone()
        x[..., h - rows:, :] = 0.0
        return x


def build_augmentation(aug: dict, image_size: int) -> v2.Compose:
    """SSL view augmentation, all parameters from config.yaml."""
    rrc = aug["random_resized_crop"]
    cj = aug["color_jitter"]
    gb = aug["gaussian_blur"]
    bm = aug["bottom_mask"]

    steps: list = [
        v2.RandomResizedCrop(image_size,
                             scale=(rrc["scale_min"], rrc["scale_max"]),
                             antialias=True),
        v2.RandomHorizontalFlip(p=aug["horizontal_flip"]),
        v2.RandomApply([v2.ColorJitter(cj["brightness"], cj["contrast"],
                                       cj["saturation"], cj["hue"])],
                       p=cj["probability"]),
        v2.RandomGrayscale(p=aug["grayscale"]),
        v2.RandomApply([v2.GaussianBlur(kernel_size=23,
                                        sigma=(gb["sigma_min"], gb["sigma_max"]))],
                       p=gb["probability"]),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
    if aug["random_erasing"] > 0:
        steps.append(v2.RandomErasing(p=aug["random_erasing"], value=0.0))
    if bm["enabled"]:
        steps.append(BottomMask(bm["probability"], bm["min_fraction"],
                                bm["max_fraction"]))
    return v2.Compose(steps)


class TwoViewDataset(Dataset):
    """Two independently augmented views of the same image."""

    def __init__(self, df: pd.DataFrame, classes: list[str], aug: dict,
                 image_size: int = 224):
        self.paths = df["path"].tolist()
        c2i = {c: i for i, c in enumerate(classes)}
        self.cities = np.asarray([c2i[c] for c in df["city_ascii"]], dtype=np.int64)
        self.tf = build_augmentation(aug, image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return self.tf(im), self.tf(im), int(self.cities[i])


def make_ssl_loader(df: pd.DataFrame, classes: list[str], cfg: dict,
                    shuffle: bool = True) -> DataLoader:
    s11 = cfg["stage11"]
    ssl = s11["ssl"]
    gen = torch.Generator()
    gen.manual_seed(cfg["project"]["seed"])
    ds = TwoViewDataset(df, classes, s11["augmentation"], ssl["image_size"])
    return DataLoader(
        ds, batch_size=ssl["batch_size"], shuffle=shuffle,
        drop_last=True,                      # VICReg's variance term needs a full batch
        num_workers=ssl["num_workers"], pin_memory=True,
        persistent_workers=ssl["num_workers"] > 0,
        worker_init_fn=_worker_init, generator=gen,
    )
