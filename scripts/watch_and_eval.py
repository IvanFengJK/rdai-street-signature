"""Run the Stage 11 evaluation automatically as target checkpoints appear.

Turns the "how many epochs is enough?" budget question into a measurement:
evaluate at several points during the run and read where retrieval quality
plateaus, rather than picking an epoch count in advance.

IMPORTANT CAVEAT, which must accompany any use of these numbers: the LR follows
a cosine schedule sized to the full run. An intermediate checkpoint is
schedule-incomplete — its learning rate is still high — so it UNDERSTATES what
a properly-completed run of that length would achieve. These points are a trend
and a lower bound, not fair standalone N-epoch models.

  python scripts/watch_and_eval.py vicreg_a 30 50 70
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO / "checkpoints" / "stage11"
POLL_SECONDS = 120


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: watch_and_eval.py <run_tag> <epoch> [epoch ...]")
    run_tag = sys.argv[1]
    targets = sorted(int(a) for a in sys.argv[2:])
    print(f"watching {CKPT_DIR}/{run_tag}_epochNNN.pt for epochs {targets}",
          flush=True)

    pending = list(targets)
    while pending:
        for ep in list(pending):
            ck = CKPT_DIR / f"{run_tag}_epoch{ep:03d}.pt"
            if not ck.exists():
                continue
            tag = f"{run_tag}_ep{ep:03d}"
            print(f"\n=== epoch {ep} checkpoint found, evaluating ===", flush=True)
            r = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "run_stage11_eval.py"),
                 "--ckpt", str(ck), "--tag", tag],
                cwd=str(REPO))
            if r.returncode != 0:
                print(f"!! evaluation FAILED for epoch {ep} "
                      f"(exit {r.returncode}) — leaving it for manual retry",
                      flush=True)
            pending.remove(ep)
        if pending:
            time.sleep(POLL_SECONDS)
    print("\nall requested checkpoints evaluated", flush=True)


if __name__ == "__main__":
    main()
