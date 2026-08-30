"""Stage 2 mop-up — KartaView only, globally rate limited.

KartaView returns HTTP 429 under sustained concurrency; the bulk pass lost
~9,100 images to it. Mapillary tolerated 32 workers fine. So this pass drives
KartaView alone through a shared token bucket, with long backoff on 429.
"""
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.imagery as IM                      # noqa: E402
from src.data import load_config              # noqa: E402

RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0   # images/sec, whole process
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6

_lock = threading.Lock()
_next = [0.0]


def _throttle():
    with _lock:
        now = time.monotonic()
        wait = max(0.0, _next[0] - now)
        _next[0] = max(now, _next[0]) + 1.0 / RATE
    if wait:
        time.sleep(wait)


_orig = IM._resolve_kartaview_once


def _paced(orig_id):
    _throttle()
    return _orig(orig_id)


IM._resolve_kartaview_once = _paced
IM.MAX_TRIES = 5            # throttles (429/409) recover; dead nodes (502) should not stall the run
IM.BACKOFF_BASE = 1.8

cfg = load_config()
root = Path(cfg["paths"]["images"])
m = pd.read_parquet("data/interim/stage2_manifest.parquet")
m = m[m.source == "KartaView"]

# Work the biggest gaps first so progress is visible early, and so a stalled
# node (storage7 is returning 502) does not hold up the rest.
todo_by_city = []
for city, g in m.groupby("city_ascii", sort=True):
    out = root / city.replace(" ", "_")
    have = {p.stem for p in out.glob("*.jpg")} if out.exists() else set()
    todo = g[~g.uuid.isin(have)]
    if len(todo):
        todo_by_city.append((city, out, todo))
todo_by_city.sort(key=lambda t: -len(t[2]))

t0 = time.time()
for city, out, todo in todo_by_city:
    res = IM.download_city(todo, out, resize_px=cfg["imagery"]["resize_px"],
                           workers=WORKERS)
    print(f"{city:16} missing={len(todo):>5} -> {res} "
          f"now={len(list(out.glob('*.jpg')))}", flush=True)
print(f"\nmop-up wall={(time.time()-t0)/60:.1f} min")
