"""Stage 11 — SSL encoder: ResNet50 backbone + VICReg projection head.

Deliberately the same timm ResNet50 family as the Stage 4 encoder, and the
representation used downstream is the same standard 2048-d pooled feature. Only
the learning objective changes, so the comparison against the Stage 5 baseline
isolates the objective rather than confounding it with architecture.

`forward_features` has the same signature as `src.model.CityEncoder`, so
`src.embed.extract` works on this model unchanged.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


def build_projector(in_dim: int, hidden: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """VICReg's expander: Linear-BN-ReLU blocks, final Linear with no BN/ReLU."""
    assert n_layers >= 2, "projector needs at least two layers"
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class SSLEncoder(nn.Module):
    def __init__(self, backbone: str, projector_cfg: dict,
                 pretrained: bool = False):
        super().__init__()
        self.backbone_name = backbone
        # num_classes=0 -> timm returns the pooled feature, not logits
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          num_classes=0)
        self.embed_dim = self.backbone.num_features
        self.projector = build_projector(self.embed_dim,
                                         projector_cfg["hidden_dim"],
                                         projector_cfg["output_dim"],
                                         projector_cfg["n_layers"])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """The representation Stage 11 evaluates — before the projector."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))


def build_ssl_model(cfg: dict) -> SSLEncoder:
    return SSLEncoder(cfg["encoder"]["backbone"], cfg["stage11"]["projector"])


def describe(model: SSLEncoder) -> str:
    bb = sum(p.numel() for p in model.backbone.parameters())
    pj = sum(p.numel() for p in model.projector.parameters())
    return (f"backbone={model.backbone_name}  embed_dim={model.embed_dim}\n"
            f"backbone params : {bb:,}\n"
            f"projector params: {pj:,}\n"
            f"total params    : {bb + pj:,}\n"
            f"projector       : {model.projector}")
