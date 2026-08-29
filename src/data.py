"""Stage 1 — build the working subset from the Global Streetscapes tables.

The HF repo ships a pre-joined master table, data/parquet/streetscapes.parquet
(9,831,714 rows x 140 cols), which already contains every column Stage 0
resolved. Reading that one file with a column projection is far cheaper than
downloading and joining simplemaps + metadata_common_attributes + segmentation
+ osm + ghsl separately (~2.7 GB).

Perception columns are never requested. See CLAUDE.md rule 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfFileSystem

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_PARQUET = "datasets/NUS-UAL/global-streetscapes/data/parquet/streetscapes.parquet"

# Columns we pull from the master table. Deliberately excludes the six
# perception columns (Beautiful, Boring, Depressing, Lively, Safe, Wealthy).
KEEP_COLUMNS = [
    # identity / join
    "uuid", "source", "orig_id",
    # split key (rule 1)
    "sequence_id", "sequence_index", "sequence_img_count",
    # city label
    "city_ascii", "city_id", "country", "continent", "city_lat", "city_lon",
    # position / capture
    "lat", "lon", "year", "month", "hour",
    "width", "height", "projection_type", "pano_status", "quality",
    # physical attributes
    "green_view_index", "sky_view_index",
    "Building", "Total", "Vegetation", "Sky",
    # road
    "highway", "type_highway", "snap_dist",
    # settlement typology
    "urban_code", "urban_term",
]

PERCEPTION_COLUMNS = ["Beautiful", "Boring", "Depressing", "Lively", "Safe", "Wealthy"]


def load_config(path: str | Path = REPO_ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def selected_city_ids(cfg: dict) -> list[int]:
    ids = [c["city_id"] for c in cfg["subset"]["cities"]]
    if not ids:
        raise ValueError("config.yaml subset.cities is empty; Stage 1 not configured")
    return ids


def build_subset(cfg: dict, out_path: Path | None = None,
                 progress: bool = True) -> pd.DataFrame:
    """Stream the master parquet row-group by row-group, keeping only the
    selected cities and the projected columns."""
    city_ids = set(selected_city_ids(cfg))
    out_path = out_path or REPO_ROOT / cfg["paths"]["interim"] / "subset.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem()
    handle = fs.open(MASTER_PARQUET, "rb")
    pf = pq.ParquetFile(handle)

    assert not (set(PERCEPTION_COLUMNS) & set(KEEP_COLUMNS)), \
        "rule 2 violation: a perception column is in the projection"

    parts = []
    n_groups = pf.metadata.num_row_groups
    for i in range(n_groups):
        tbl = pf.read_row_group(i, columns=KEEP_COLUMNS)
        df = tbl.to_pandas()
        df = df[df.city_id.isin(city_ids)]
        if len(df):
            parts.append(df)
        if progress:
            kept = sum(len(p) for p in parts)
            print(f"  row group {i+1}/{n_groups}  kept={kept:,}", flush=True)

    sub = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEEP_COLUMNS)
    sub.to_parquet(out_path, index=False)
    return sub


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add building_view_index. Stage 0 established that no building-density
    column ships with the dataset; this is Building / Total, by analogy with
    the shipped green_view_index (= Vegetation / Total)."""
    df = df.copy()
    total = df["Total"].where(df["Total"] > 0)
    df["building_view_index"] = df["Building"] / total
    return df
