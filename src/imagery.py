"""Stage 2 — download street imagery and pre-resize to 256 px on disk.

Two sources, with very different access requirements:

  KartaView  — public JSON API (api.kartaview.org / api.openstreetcam.org),
               no credentials. Resolve orig_id -> fileurlProc, then GET.
  Mapillary  — requires a personal API access token. The upstream project's
               download_jpegs_mapillary.py hardcodes
               `access_token = 'INSERT-YOUR-TOKEN-HERE'`. Without a token the
               Graph API returns 401 and no image can be fetched.

Set MAPILLARY_TOKEN in the environment to enable the Mapillary path.

Idempotent: an image whose destination file already exists is skipped, so the
job can be interrupted and resumed freely.
"""

from __future__ import annotations

import io
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

KV_API = "https://api.kartaview.org/2.0/photo/?id={orig_id}"
MLY_API = "https://graph.mapillary.com/{orig_id}?fields=thumb_2048_url&access_token={token}"

_session = requests.Session()
_session.headers.update({"User-Agent": "parallel-streets/0.1 (course project)"})

# Both APIs rate-limit under concurrency. A first pass at 32 workers lost 3,120
# Jakarta images to transient HTTP errors that succeed immediately when retried,
# so every network call goes through this backoff.
MAX_TRIES = 4
BACKOFF_BASE = 1.7


def _with_retry(fn, *a, **kw):
    last = None
    for attempt in range(MAX_TRIES):
        try:
            return fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            # 409 is Azure's BlobArchived: KartaView has moved the image to
            # archive-tier storage and it cannot be served at all. 410/404 are
            # gone, 401/403 unauthorised. None of these are worth retrying.
            if status is not None and status in (401, 403, 404, 409, 410):
                raise
            if attempt == MAX_TRIES - 1:
                break
            sleep = BACKOFF_BASE ** attempt + random.random()
            if status == 429:
                retry_after = getattr(e, "response", None)
                ra = (retry_after.headers.get("Retry-After") if retry_after is not None
                      else None)
                sleep = max(sleep, float(ra) if ra and ra.isdigit() else 5.0)
            time.sleep(sleep)
    raise last


def mapillary_token() -> str | None:
    return os.environ.get("MAPILLARY_TOKEN") or os.environ.get("MLY_TOKEN")


def _resolve_kartaview(orig_id) -> str | None:
    return _with_retry(_resolve_kartaview_once, orig_id)


def _resolve_kartaview_once(orig_id) -> str | None:
    r = _session.get(KV_API.format(orig_id=orig_id), timeout=30)
    r.raise_for_status()
    data = r.json().get("result")
    if not data or not data.get("data"):
        return None
    return data["data"][0].get("fileurlProc") or data["data"][0].get("fileurl")


def _resolve_mapillary(orig_id, token: str) -> str | None:
    return _with_retry(_resolve_mapillary_once, orig_id, token)


def _resolve_mapillary_once(orig_id, token: str) -> str | None:
    r = _session.get(MLY_API.format(orig_id=orig_id, token=token), timeout=30)
    if r.status_code == 401:
        raise PermissionError("Mapillary rejected the access token (401)")
    r.raise_for_status()
    return r.json().get("thumb_2048_url")


def fetch_one(row, out_dir: Path, resize_px: int, token: str | None) -> str:
    """Returns one of: 'skipped', 'ok', 'no_url', 'error:<msg>'."""
    dst = out_dir / f"{row.uuid}.jpg"
    if dst.exists():
        return "skipped"
    try:
        if row.source == "KartaView":
            url = _resolve_kartaview(row.orig_id)
        else:
            if not token:
                return "error:no_mapillary_token"
            url = _resolve_mapillary(row.orig_id, token)
        if not url:
            return "no_url"
        def _get_decode():
            resp = _session.get(url, timeout=90)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")

        im = _with_retry(_get_decode)
        # pre-resize: shortest side to resize_px, preserving aspect ratio
        w, h = im.size
        scale = resize_px / min(w, h)
        if scale < 1:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                           Image.BICUBIC)
        tmp = dst.with_suffix(".tmp")
        im.save(tmp, "JPEG", quality=90)
        tmp.rename(dst)          # atomic: a partial file is never seen as done
        return "ok"
    except Exception as e:  # noqa: BLE001 - report, never silently drop
        return f"error:{type(e).__name__}:{e}"[:200]


def download_city(df: pd.DataFrame, out_dir: Path, resize_px: int = 256,
                  workers: int = 32, token: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    token = token if token is not None else mapillary_token()
    counts: dict[str, int] = {}
    t0 = time.time()
    rows = list(df.itertuples())
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_one, r, out_dir, resize_px, token) for r in rows]
        for f in as_completed(futs):
            res = f.result()
            key = res if not res.startswith("error:") else "error:" + res.split(":")[1]
            counts[key] = counts.get(key, 0) + 1
    counts["elapsed_s"] = round(time.time() - t0, 1)
    return counts


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*.jpg"))
