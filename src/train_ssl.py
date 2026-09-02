"""Stage 11 — SSL training loop.

Preserves the Stage 5 reproducibility standard: seed everything before the
model is built, seed DataLoader workers, checkpoint every epoch with full
optimizer / scheduler / scaler state so a run is resumable and idempotent.

Experiment A uses the VICReg loss alone. The adversarial branch is added in
Experiment B; this loop takes an optional `adversary` argument so B does not
require rewriting it, but with `adversarial.enabled: false` nothing here
touches city labels.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast

from .vicreg import vicreg_loss


def cosine_warmup(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * p))


def _ckpt_path(out_dir: Path, tag: str, epoch: int | None = None) -> Path:
    return out_dir / (f"{tag}_last.pt" if epoch is None
                      else f"{tag}_epoch{epoch:03d}.pt")


@torch.no_grad()
def evaluate_ssl(model, loader, cfg, device) -> dict:
    """VICReg loss over the val split. Monitoring only — never used for model
    selection, so it cannot leak val information into the representation."""
    v = cfg["stage11"]["vicreg"]
    model.eval()
    sums: dict[str, float] = {}
    n = 0
    for xa, xb, _ in loader:
        xa = xa.to(device, non_blocking=True)
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda", dtype=torch.float16, enabled=cfg["stage11"]["ssl"]["amp"]):
            za, zb = model(xa), model(xb)
        _, parts = vicreg_loss(za, zb, v["sim_coeff"], v["std_coeff"],
                               v["cov_coeff"], v["eps"])
        for k, val in parts.items():
            sums[k] = sums.get(k, 0.0) + val
        n += 1
    model.train()
    return {f"val_{k}": s / max(1, n) for k, s in sums.items()}


def train_ssl(model, train_loader, val_loader, cfg: dict, tag: str,
              device: str = "cuda", adversary=None, resume: bool = True):
    """Train with VICReg. Returns the epoch history.

    `adversary`, when supplied by Experiment B, must be a callable taking
    (features, city_labels) and returning (loss, metrics_dict).
    """
    s11 = cfg["stage11"]
    ssl_cfg = s11["ssl"]
    v = s11["vicreg"]
    out_dir = Path(s11["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    params = list(model.parameters())
    if adversary is not None:
        params += list(adversary.parameters())
    opt = torch.optim.AdamW(params, lr=ssl_cfg["lr"],
                            weight_decay=ssl_cfg["weight_decay"])
    scaler = GradScaler("cuda", enabled=ssl_cfg["amp"])

    epochs = ssl_cfg["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    start_epoch, history, gstep = 0, [], 0
    last = _ckpt_path(out_dir, tag)
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        if adversary is not None and ck.get("adversary") is not None:
            adversary.load_state_dict(ck["adversary"])
        start_epoch = ck["epoch"]
        history = ck["history"]
        gstep = ck["global_step"]
        print(f"resumed {tag} from epoch {start_epoch}", flush=True)

    model.train()
    for ep in range(start_epoch, epochs):
        t0 = time.monotonic()   # monotonic: this host's wall clock jumps under WSL
        acc: dict[str, float] = {}
        nb = 0
        for xa, xb, city in train_loader:
            lr = cosine_warmup(gstep, total_steps,
                               ssl_cfg["warmup_epochs"] * steps_per_epoch,
                               ssl_cfg["lr"])
            for g in opt.param_groups:
                g["lr"] = lr

            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.float16, enabled=ssl_cfg["amp"]):
                fa = model.backbone(xa)
                fb = model.backbone(xb)
                za = model.projector(fa)
                zb = model.projector(fb)

            loss, parts = vicreg_loss(za, zb, v["sim_coeff"], v["std_coeff"],
                                      v["cov_coeff"], v["eps"])
            if adversary is not None:
                city = city.to(device, non_blocking=True)
                adv_loss, adv_parts = adversary(torch.cat([fa, fb]).float(),
                                                torch.cat([city, city]))
                loss = loss + adv_loss
                parts.update(adv_parts)
                parts["total"] = loss.item()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            for k, val in parts.items():
                acc[k] = acc.get(k, 0.0) + val
            nb += 1
            gstep += 1

            every = ssl_cfg.get("log_every_steps", 0)
            if every and nb % every == 0:
                el = time.monotonic() - t0
                ips = nb * xa.size(0) * 2 / el          # two views per image
                eta = (steps_per_epoch - nb) * el / nb
                print(f"  ep{ep+1:>3} step {nb:>4}/{steps_per_epoch}  "
                      f"total={parts['total']:.3f}  std={parts['std_mean']:.3f}  "
                      f"{ips:.0f} img/s  epoch ETA {eta/60:.1f} min", flush=True)

        row = {"epoch": ep + 1, "lr": lr, "secs": round(time.monotonic() - t0, 1)}
        row.update({k: s / max(1, nb) for k, s in acc.items()})

        if val_loader is not None and ((ep + 1) % ssl_cfg["val_every"] == 0
                                       or ep + 1 == epochs):
            row.update(evaluate_ssl(model, val_loader, cfg, device))

        history.append(row)
        msg = (f"epoch {ep+1:>3}/{epochs}  total={row['total']:.4f}  "
               f"inv={row['invariance']:.4f}  var={row['variance']:.4f}  "
               f"cov={row['covariance']:.4f}  std={row['std_mean']:.3f}")
        if "adv_acc" in row:
            msg += f"  adv_acc={row['adv_acc']:.3f}"
        if "val_total" in row:
            msg += f"  | val={row['val_total']:.4f}"
        print(msg + f"  ({row['secs']:.0f}s)", flush=True)

        state = {
            "epoch": ep + 1, "tag": tag, "global_step": gstep,
            "model": model.state_dict(),
            "adversary": adversary.state_dict() if adversary is not None else None,
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": {"type": "cosine_warmup", "base_lr": ssl_cfg["lr"],
                          "warmup_steps": ssl_cfg["warmup_epochs"] * steps_per_epoch,
                          "total_steps": total_steps},
            "cfg": cfg, "seed": cfg["project"]["seed"], "history": history,
        }
        torch.save(state, _ckpt_path(out_dir, tag))              # resume point
        torch.save(state, _ckpt_path(out_dir, tag, ep + 1))      # every epoch

    json.dump({"history": history, "cfg": cfg["stage11"],
               "seed": cfg["project"]["seed"]},
              open(out_dir / f"{tag}_history.json", "w", encoding="utf-8"), indent=2)
    return history
