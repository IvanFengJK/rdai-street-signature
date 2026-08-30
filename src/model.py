"""Stage 4 — the encoder.

timm backbone with a replaced head; city classification is the pretext task.
Config-switchable between ResNet50 and a small ViT so the two can be compared.

pretrained=False by default: the point of the project is that this model was
trained here, not downloaded.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CityEncoder(nn.Module):
    """Backbone + linear city head. `forward_features` gives the embedding
    Stage 6 caches after the head is stripped."""

    def __init__(self, backbone: str, n_classes: int, pretrained: bool = False,
                 drop_rate: float = 0.0):
        super().__init__()
        self.backbone_name = backbone
        # num_classes=0 makes timm return pooled features instead of logits
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        self.embed_dim = self.backbone.num_features
        self.head = nn.Linear(self.embed_dim, n_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_model(cfg: dict, n_classes: int) -> CityEncoder:
    enc = cfg["encoder"]
    return CityEncoder(enc["backbone"], n_classes,
                       pretrained=enc.get("pretrained", False),
                       drop_rate=enc.get("drop_rate", 0.0))


def describe(model: CityEncoder) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (f"backbone={model.backbone_name}  embed_dim={model.embed_dim}\n"
            f"total params    : {total:,}\n"
            f"trainable params: {trainable:,}\n"
            f"head            : {model.head}")
