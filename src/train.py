"""Stage 5 — training loop. Mixed precision, checkpoint every epoch."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast


def set_seed(seed: int) -> None:
    """Full determinism so 'train twice from the same seed' reproduces."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, n_classes):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = crit(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            for t, p in zip(y.cpu(), pred.cpu()):
                cm[t, p] += 1
    return loss_sum / total, correct / total, cm


def train(model, train_loader, val_loader, cfg, classes, device="cuda",
          ckpt_dir: Path | None = None, tag: str = "run"):
    set_seed(cfg["project"]["seed"])
    tc = cfg["train"]
    ckpt_dir = Path(ckpt_dir or cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=tc.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"],
                            weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tc["lr"], epochs=tc["epochs"],
        steps_per_epoch=len(train_loader))
    scaler = GradScaler("cuda", enabled=tc["amp"])

    history = []
    best = 0.0
    for ep in range(tc["epochs"]):
        model.train()
        t0 = time.time()
        run_loss = seen = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=tc["amp"], dtype=torch.float16):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * y.size(0)
            seen += y.size(0)
        tr_loss = run_loss / seen
        va_loss, va_acc, cm = evaluate(model, val_loader, device, len(classes))
        history.append({"epoch": ep + 1, "train_loss": tr_loss,
                        "val_loss": va_loss, "val_acc": va_acc,
                        "secs": round(time.time() - t0, 1)})
        print(f"epoch {ep+1:>2}/{tc['epochs']}  train_loss={tr_loss:.4f}  "
              f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  "
              f"({history[-1]['secs']:.0f}s)", flush=True)

        torch.save({"epoch": ep + 1, "model": model.state_dict(),
                    "classes": classes, "cfg": cfg, "val_acc": va_acc},
                   ckpt_dir / f"{tag}_epoch{ep+1:02d}.pt")
        if va_acc > best:
            best = va_acc
            torch.save({"epoch": ep + 1, "model": model.state_dict(),
                        "classes": classes, "cfg": cfg, "val_acc": va_acc},
                       ckpt_dir / f"{tag}_best.pt")

    json.dump({"history": history, "confusion_matrix": cm.tolist(),
               "classes": classes, "best_val_acc": best},
              open(ckpt_dir / f"{tag}_history.json", "w"), indent=2)
    return history, cm, best
