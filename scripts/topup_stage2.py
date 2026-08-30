"""Stage 2 top-up — replace permanently unavailable images.

Part of KartaView's archive is on Azure archive-tier storage and returns
HTTP 409 BlobArchived: those images cannot be served to anyone, and no amount
of retrying changes that. A few are on a storage node returning 502.

The Stage 2 target is ~5,000 images per city; *which* eligible images is a
sampling detail. So for each city still short, draw replacements from that
city's unused eligible pool (same flat perspective/fisheye filter, same
sequence-stratified sampling) and fetch those instead.

Mapillary is preferred for replacements because it has proved reliable, while
KartaView is where the archived blobs are. The resulting per-city source mix is
reported so the shift is visible rather than silent.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import add_derived, flat_only, load_config, sample_per_city  # noqa: E402
from src.imagery import download_city, mapillary_token                     # noqa: E402

TARGET = None          # from config
ROUNDS = 4             # replacements can themselves fail; iterate

cfg = load_config()
TARGET = cfg["imagery"]["images_per_city"]
seed = cfg["project"]["seed"]
root = Path(cfg["paths"]["images"])

if not mapillary_token():
    sys.exit("MAPILLARY_TOKEN not set. Aborting.")

full = add_derived(pd.read_parquet("data/interim/subset_clean.parquet"))
pool = flat_only(full)
manifest = pd.read_parquet("data/interim/stage2_manifest.parquet")

tried: set[str] = set(manifest.uuid)      # never re-attempt a known-dead uuid

for rnd in range(ROUNDS):
    shortfalls = {}
    for city in sorted(pool.city_ascii.unique()):
        d = root / city.replace(" ", "_")
        have = len(list(d.glob("*.jpg"))) if d.exists() else 0
        if have < TARGET:
            shortfalls[city] = TARGET - have
    if not shortfalls:
        print("all cities complete")
        break
    print(f"\n=== round {rnd+1}: {len(shortfalls)} cities short, "
          f"{sum(shortfalls.values()):,} images ===", flush=True)

    for city, need in sorted(shortfalls.items(), key=lambda kv: -kv[1]):
        cand = pool[(pool.city_ascii == city) & (~pool.uuid.isin(tried))]
        if cand.empty:
            print(f"{city:16} need={need:>5} but pool exhausted", flush=True)
            continue
        # prefer Mapillary: KartaView is where the archived blobs are
        mly = cand[cand.source == "Mapillary"]
        kv = cand[cand.source == "KartaView"]
        take = min(len(cand), int(need * 1.6) + 40)     # over-draw for failures
        pick = pd.concat([mly, kv]).head(take) if len(mly) else cand.head(take)
        pick = sample_per_city(pick, min(take, len(pick)), seed + rnd)
        tried.update(pick.uuid)
        res = download_city(pick, root / city.replace(" ", "_"),
                            resize_px=cfg["imagery"]["resize_px"], workers=16)
        now = len(list((root / city.replace(" ", "_")).glob("*.jpg")))
        print(f"{city:16} need={need:>5} drew={len(pick):>5} -> {res} now={now}",
              flush=True)

# final audit
rows = []
for city in sorted(pool.city_ascii.unique()):
    d = root / city.replace(" ", "_")
    files = {p.stem for p in d.glob("*.jpg")} if d.exists() else set()
    src = full[full.uuid.isin(files)].source.value_counts()
    rows.append({"city": city, "images": len(files),
                 "mapillary": int(src.get("Mapillary", 0)),
                 "kartaview": int(src.get("KartaView", 0))})
rep = pd.DataFrame(rows).set_index("city")
print("\n" + rep.to_string())
print(f"\nTOTAL {rep.images.sum():,} images; cities at target: "
      f"{(rep.images >= TARGET).sum()}/{len(rep)}")
rep.to_csv("data/interim/stage2_final_counts.csv")
