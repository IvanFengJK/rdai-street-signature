"""Stage 2 fill pass — retry whatever the first pass missed.

Lower concurrency plus the retry/backoff added to src/imagery.py after the
first pass showed transient rate-limit failures (Jakarta lost 3,120 images to
HTTP errors that succeed on retry).
"""
import json, sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_config                                   # noqa: E402
from src.imagery import download_city, dir_size_bytes, mapillary_token  # noqa: E402

cfg = load_config()
root = Path(cfg["paths"]["images"])
manifest = pd.read_parquet("data/interim/stage2_manifest.parquet")
if not mapillary_token():
    sys.exit("MAPILLARY_TOKEN not set. Aborting.")

t0 = time.time()
summary = {}
for city, g in manifest.groupby("city_ascii", sort=True):
    out = root / city.replace(" ", "_")
    have = {p.stem for p in out.glob("*.jpg")}
    todo = g[~g.uuid.isin(have)]
    if todo.empty:
        print(f"{city:16} complete ({len(have)})", flush=True)
        summary[city] = {"ok": 0, "files_on_disk": len(have)}
        continue
    res = download_city(todo, out, resize_px=cfg["imagery"]["resize_px"], workers=12)
    res["files_on_disk"] = len(list(out.glob("*.jpg")))
    res["bytes"] = dir_size_bytes(out)
    summary[city] = res
    print(f"{city:16} missing={len(todo):>5} -> {res}", flush=True)

total = sum(v["files_on_disk"] for v in summary.values())
tb = sum(v.get("bytes", 0) for v in summary.values())
print(f"\nAFTER FILL: files={total:,} size={tb/1e9:.2f} GB wall={(time.time()-t0)/60:.1f} min")
json.dump(summary, open("data/interim/stage2_fill_report.json", "w"), indent=2)
