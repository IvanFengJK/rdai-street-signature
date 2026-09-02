"""Stage 11 — VICReg loss.

Bardes, Ponce & LeCun, "VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning" (ICLR 2022).

Three terms on the projector outputs of two views:

  invariance  - MSE between the two views' embeddings; pulls views of the same
                image together.
  variance    - hinge on the per-dimension standard deviation across the batch,
                keeping it above 1. This is what prevents collapse: without it
                the invariance term alone is minimised by a constant output.
  covariance  - sum of squared off-diagonal covariances, decorrelating the
                dimensions so they do not all encode the same thing.

The loss is computed in float32 even under AMP: the covariance term squares a
D x D matrix and the variance term takes a square root, both of which lose
meaningful precision in fp16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m, "covariance matrix must be square"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_loss(z: torch.Tensor, eps: float) -> torch.Tensor:
    """Hinge keeping each dimension's batch std above 1."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Squared off-diagonal covariance, normalised by dimensionality."""
    n, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n - 1)
    return _off_diagonal(cov).pow(2).sum() / d


def vicreg_loss(za: torch.Tensor, zb: torch.Tensor, sim_coeff: float,
                std_coeff: float, cov_coeff: float, eps: float = 1e-4
                ) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns (total_loss, component_dict). Components are detached floats,
    logged per epoch so a collapse is visible in the history rather than only
    in the final number."""
    za = za.float()
    zb = zb.float()

    inv = F.mse_loss(za, zb)
    var = 0.5 * (variance_loss(za, eps) + variance_loss(zb, eps))
    cov = 0.5 * (covariance_loss(za) + covariance_loss(zb))

    total = sim_coeff * inv + std_coeff * var + cov_coeff * cov
    parts = {
        "invariance": inv.item(),
        "variance": var.item(),
        "covariance": cov.item(),
        # Mean per-dimension std. Collapse shows up here first: it falls
        # toward 0 while the variance hinge saturates at 1.
        "std_mean": torch.sqrt(za.var(dim=0) + eps).mean().item(),
        "total": total.item(),
    }
    return total, parts
