"""Stage 2 runner — download the manifest, resize to 256 px, report size + time."""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config          # noqa: E402
from src.imagery import download_city, dir_size_bytes, mapillary_token  # noqa: E402

cfg = load_config()
root = Path(cfg["paths"]["images"])
manifest = pd.read_parquet("data/interim/stage2_manifest.parquet")

if not mapillary_token():
    sys.exit("MAPILLARY_TOKEN not set; 86% of the manifest is Mapillary. Aborting.")

t0 = time.time()
summary = {}
for city, g in manifest.groupby("city_ascii", sort=True):
    out = root / city.replace(" ", "_")
    res = download_city(g, out, resize_px=cfg["imagery"]["resize_px"], workers=32)
    res["files_on_disk"] = len(list(out.glob("*.jpg")))
    res["bytes"] = dir_size_bytes(out)
    summary[city] = res
    print(f"{city:16} {res}", flush=True)

total_files = sum(v["files_on_disk"] for v in summary.values())
total_bytes = sum(v["bytes"] for v in summary.values())
wall = time.time() - t0
print(f"\nTOTAL files={total_files:,} size={total_bytes/1e9:.2f} GB "
      f"wall={wall/60:.1f} min")
Path("data/interim").mkdir(parents=True, exist_ok=True)
json.dump({"per_city": summary, "total_files": total_files,
           "total_bytes": total_bytes, "wall_s": round(wall, 1)},
          open("data/interim/stage2_report.json", "w"), indent=2)
