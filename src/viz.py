"""Stages 8-9 — UMAP projection and Grad-CAM.

Static matplotlib only. No web UI, no interactive globe, no Gradio (rule 6).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ---------------------------------------------------------------- Stage 8
def umap_project(emb: np.ndarray, seed: int, n_neighbors: int = 30,
                 min_dist: float = 0.1) -> np.ndarray:
    import umap
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                     metric="cosine", random_state=seed).fit_transform(emb)


def plot_umap(xy: np.ndarray, df: pd.DataFrame, out: Path,
              greenery_col: str = "green_view_index") -> Path:
    """Two panels: coloured by city, then recoloured by greenery."""
    fig, axes = plt.subplots(1, 2, figsize=(19, 8.5))
    cities = sorted(df["city_ascii"].unique())
    cmap = plt.get_cmap("tab20", len(cities))
    for i, c in enumerate(cities):
        m = (df["city_ascii"] == c).to_numpy()
        axes[0].scatter(xy[m, 0], xy[m, 1], s=2, alpha=0.5,
                        color=cmap(i), label=c, linewidths=0)
    axes[0].set_title("UMAP of the trained embedding — coloured by city")
    axes[0].legend(markerscale=5, fontsize=7, ncol=2, loc="best", framealpha=0.9)

    sc = axes[1].scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.6,
                         c=df[greenery_col].to_numpy(), cmap="YlGn",
                         vmin=0, vmax=float(np.nanpercentile(df[greenery_col], 98)),
                         linewidths=0)
    axes[1].set_title("Same projection — recoloured by greenery (green view index)")
    fig.colorbar(sc, ax=axes[1], label=greenery_col, shrink=0.8)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- Stage 5 plots
def plot_history(history: list[dict], out: Path) -> Path:
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(h.epoch, h.train_loss, marker="o", label="train")
    ax[0].plot(h.epoch, h.val_loss, marker="o", label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy loss")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(h.epoch, h.val_acc, marker="o", color="tab:green")
    ax[1].axhline(1/20, ls="--", c="grey", label="chance (1/20)")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val accuracy")
    ax[1].set_title("Validation accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def plot_confusion(cm, classes, out: Path) -> Path:
    cm = np.asarray(cm, dtype=float)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(cmn, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalised)")
    fig.colorbar(im, shrink=0.8); fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


# ---------------------------------------------------------------- Stage 7 map
def plot_pairs_map(df: pd.DataFrame, q_idx, nb, out: Path) -> Path:
    """Static world map: query -> retrieved doppelganger in another city."""
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.scatter(df.lon, df.lat, s=0.5, c="0.85", linewidths=0)
    for qi, row in zip(q_idx, nb):
        for n in row[:1]:
            if n < 0:
                continue
            ax.plot([df.lon.iloc[qi], df.lon.iloc[n]],
                    [df.lat.iloc[qi], df.lat.iloc[n]],
                    lw=0.8, alpha=0.75, c="tab:red")
    ax.scatter(df.lon.iloc[q_idx], df.lat.iloc[q_idx], s=22, c="tab:blue",
               zorder=3, label="query")
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Street doppelgangers — query to top cross-city match")
    ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


def plot_pair_cards(df: pd.DataFrame, q_idx, nb, out: Path, n: int = 6) -> Path:
    """Paired image cards: query beside its top cross-city neighbour."""
    from PIL import Image
    n = min(n, len(q_idx))
    fig, axes = plt.subplots(n, 2, figsize=(7.5, 3.1 * n))
    axes = np.atleast_2d(axes)
    for r in range(n):
        qi = q_idx[r]; ni = nb[r][0]
        for c, i in enumerate([qi, ni]):
            a = axes[r, c]; a.set_xticks([]); a.set_yticks([])
            if i < 0:
                a.text(.5, .5, "no cross-city match", ha="center"); continue
            a.imshow(Image.open(df.iloc[i]["path"]).convert("RGB"))
            row = df.iloc[i]
            a.set_title(f"{'QUERY' if c==0 else 'MATCH'} · {row.city_ascii}\n"
                        f"gvi={row.green_view_index:.2f} bvi={row.building_view_index:.2f}",
                        fontsize=8)
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ---------------------------------------------------------------- Stage 9
def gradcam_panel(model, df: pd.DataFrame, cfg: dict, out: Path,
                  n: int = 8, device: str = "cuda") -> Path:
    """Grad-CAM over the city head.

    The honest question this answers: is the encoder reading the streetscape,
    or is it reading licence plates, road markings, vehicle shapes and camera
    furniture? This panel stays in the notebook whatever it shows.
    """
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from PIL import Image
    from .dataset import StreetDataset

    classes = sorted(df["city_ascii"].unique())
    model = model.to(device).eval()
    if "resnet" in model.backbone_name:
        targets_layers = [model.backbone.layer4[-1]]
    else:
        targets_layers = [model.backbone.blocks[-1].norm1]

    ds = StreetDataset(df.head(n), classes, train=False,
                       image_size=cfg["dataloader"]["image_size"])
    cam = GradCAM(model=model, target_layers=targets_layers)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.6))
    for i in range(n):
        x, y = ds[i]
        g = cam(input_tensor=x.unsqueeze(0).to(device),
                targets=[ClassifierOutputTarget(y)])[0]
        raw = Image.open(df.iloc[i]["path"]).convert("RGB").resize(
            (cfg["dataloader"]["image_size"],) * 2)
        axes[0, i].imshow(raw); axes[0, i].set_title(df.iloc[i]["city_ascii"], fontsize=8)
        axes[1, i].imshow(raw); axes[1, i].imshow(g, cmap="jet", alpha=0.5)
        for a in (axes[0, i], axes[1, i]):
            a.set_xticks([]); a.set_yticks([])
    axes[0, 0].set_ylabel("image", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
    fig.suptitle("Stage 9 — what the encoder attends to when predicting city")
    fig.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    return out
