# Parallel Streets — working notes

Running log for stages completed. Written as I go, per the instruction to
record findings here instead of stopping to report.

**Status: halted at Stage 2.** Stages 0 and 1 passed. Stage 2's acceptance
criterion cannot be met without a Mapillary API token. Details in the Stage 2
section. Stages 3–10 are not attempted, because every one of them consumes the
imagery Stage 2 was supposed to produce.

---

## Stage 0 — schema ✅

Fetched `info.csv` (410 rows, a data dictionary of file → field → explanation)
and, rather than trusting it alone, verified every column against the live CSV
headers on HuggingFace via HTTP range requests. Full record in `schema.json`.

| Role | File | Column |
|---|---|---|
| city | `data/simplemaps.csv` | `city_ascii` (+ `city_id`, `country`, `continent`) |
| sequence ID | `data/metadata_common_attributes.csv` | `sequence_id` |
| greenery | `data/segmentation.csv` | `green_view_index` (shipped, = `Vegetation`/`Total`) |
| building density | `data/segmentation.csv` | **none — derived**, `Building`/`Total` |
| road type | `data/osm.csv` | `highway`, `type_highway` |

Join key is `uuid`, present in every table.

### Assumptions flagged at Stage 0

1. **Building density is not a dataset column.** Nothing named building density
   ships. I derive `building_view_index = Building / Total`, exactly by analogy
   with the shipped `green_view_index = Vegetation / Total`. It measures the
   building share *of the image*, not built-up area per unit ground — a visual
   density proxy, not the planning definition. `src/schema.py:column_for()`
   raises for this role rather than returning a name, so a derived field cannot
   be silently mistaken for a dataset field.
2. **`sequence_id` uniqueness was unverified.** Resolved at Stage 1 — see below.
3. **City attribution is SimpleMaps nearest-city matching**, not administrative
   boundaries, so `city_id` is the grouping key and `city_ascii` only a label.
4. **OSM attribution is a spatial snap.** Resolved at Stage 1 — see below.

### Corrections to CLAUDE.md's own text

- The perception columns are actually `Safe, Lively, Beautiful, Wealthy,
  Boring, Depressing` — capitalised adjectives, not `safety/wealth/beauty/…`.
  Recorded under `forbidden` in `schema.json` so they are dropped by real name.
  They are never projected out of the parquet at all (`src/data.py` asserts it).
- `info.csv` has a documentation bug: it lists `places365.csv`'s fourth field as
  `Safe` (copy-paste from `perception.csv`). The real header is `place`.

### Pinned versions

```
torch==2.10.0   torchvision==0.25.0   timm==1.0.26
```

I cannot execute a Colab runtime, so this is inferred from evidence:
colabtools#5801 announced torch 2.10.0 / torchvision 0.25.0 and a Colab
maintainer confirmed *in-runtime* on 2026-03-23 that `torch.__version__`
printed `2.10.0`; the issue closed the next day. The three later upgrade
requests — #5925 (2.11), #5982 (2.12.1), #6053 (2.13) — are all still **open**,
i.e. announced by the PyTorch team but not adopted by Colab. Colab Python 3.12,
CUDA 12.5. timm is not preinstalled on Colab, so there is no shipped version to
match; 1.0.26 is the release contemporaneous with that verified snapshot.

Everything else in `requirements.txt` is a floor, not a pin: the public Colab
runtime image mirror has been stale since 2025-09 and the release notes need
sign-in, so pinning numpy/pandas from guesswork would risk pip resolving its
way into a torch downgrade. **Confirm before Stage 10** with
`import torch, torchvision; print(torch.__version__, torchvision.__version__)`.

Local install resolved to `torch 2.10.0+cu128` and CUDA works on the RTX 3080 Ti.

---

## Stage 1 — subset ✅ (acceptance met)

### A significant find

`data/parquet/streetscapes.parquet` (1,984 MB, 9,831,714 × 140) is a
**pre-joined master table** already containing every column Stage 0 resolved —
city, sequence, segmentation, OSM, GHSL. Reading that one file with a column
projection (33 of 140 columns) replaces downloading and joining five separate
tables (~2.7 GB). This is not mentioned in CLAUDE.md and is worth knowing.

The six perception columns live in this table too. `src/data.py` never lists
them in the projection and asserts that none appear in it.

Wall time: **21.4 min** to stream all 115 row groups and filter. Result:
**2,116,029 rows**, cached to `data/interim/subset.parquet`.

### Cities

20 cities, all six continents, chosen for distinct Köppen climate zones, matched
on `city_id` because city name strings are not globally unique (verified: no
name collisions among these 20 in the full 688).

| city | country | continent | images | sequences | KartaView | KartaView % | panoramas | mean GVI | mean BVI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Berlin | Germany | Europe | 277,745 | 2,076 | 82,849 | 29.8 | 6,503 | 0.082 | 0.222 |
| Sao Paulo | Brazil | South America | 271,892 | 2,247 | 77,890 | 28.6 | 0 | 0.164 | 0.209 |
| Washington | United States | North America | 203,378 | 2,908 | 10,023 | 4.9 | 54,936 | 0.091 | 0.226 |
| Taipei | Taiwan | Asia | 194,522 | 1,043 | 12 | 0.0 | 2,430 | 0.127 | 0.266 |
| Moscow | Russia | Europe | 192,633 | 3,143 | 5,146 | 2.7 | 2,126 | 0.035 | 0.281 |
| Lima | Peru | South America | 116,396 | 292 | 0 | 0.0 | 94,889 | 0.029 | 0.199 |
| Jakarta | Indonesia | Asia | 115,392 | 1,003 | 85,303 | 73.9 | 821 | 0.272 | 0.116 |
| Athens | Greece | Europe | 104,294 | 599 | 114 | 0.1 | 71,120 | 0.072 | 0.278 |
| Melbourne | Australia | Oceania | 90,149 | 933 | 12,591 | 14.0 | 0 | 0.188 | 0.250 |
| Lisbon | Portugal | Europe | 85,895 | 573 | 4 | 0.0 | 1,420 | 0.056 | 0.310 |
| Ottawa | Canada | North America | 68,099 | 749 | 20,943 | 30.8 | 1,056 | 0.142 | 0.185 |
| San Jose | Costa Rica | North America | 67,304 | 950 | 23,459 | 34.9 | 2,555 | 0.101 | 0.215 |
| San Francisco | United States | North America | 66,785 | 320 | 5,138 | 7.7 | 2,052 | 0.207 | 0.200 |
| Tokyo | Japan | Asia | 63,812 | 839 | 2,882 | 4.5 | 6,960 | 0.073 | 0.296 |
| Kuwait City | Kuwait | Asia | 62,640 | 733 | 0 | 0.0 | 152 | 0.075 | 0.191 |
| Maseru | Lesotho | Africa | 33,225 | 446 | 6,000 | 18.1 | 0 | 0.208 | 0.055 |
| Kampala | Uganda | Africa | 30,677 | 646 | 425 | 1.4 | 1,303 | 0.190 | 0.160 |
| Dar es Salaam | Tanzania | Africa | 29,762 | 219 | 0 | 0.0 | 116 | 0.223 | 0.207 |
| Singapore | Singapore | Asia | 22,327 | 204 | 9,977 | 44.7 | 0 | 0.439 | 0.069 |
| Casablanca | Morocco | Africa | 18,672 | 9,641 | 487 | 2.6 | 9,530 | 0.050 | 0.338 |

**Acceptance: no city under 2,000 images — PASS.** Smallest is Casablanca at
18,672, nearly 10× the threshold.

430 rows with null `green_view_index` were dropped (0.02%). Clean subset:
**2,115,599 rows** in `data/interim/subset_clean.parquet`.

### Assumptions resolved

- **`sequence_id` is safe as the split key on its own.** Zero sequence_ids span
  more than one source, and zero span more than one city, across 29,564
  sequences. The `(source, sequence_id)` compound key I flagged as possibly
  necessary at Stage 0 is not needed.
- **`snap_dist` is already capped at 10 m upstream** (max exactly 10.00), so the
  20 m threshold I put in `config.yaml` is a no-op. Left in as documentation.
  5.51% of rows have no OSM join at all (null `highway`/`type_highway`).

### New observations that will matter later

- **Sequence density is wildly uneven, and this is a real risk to Stage 3.**
  Lima has 292 sequences for 116,396 images (~399 images/sequence); San
  Francisco 320 for 66,785 (~209); Casablanca 9,641 for 18,672 (~2). A split by
  sequence ID is coarse-grained in the low-sequence cities — one Lima sequence
  is 0.34% of that city. The split will need stratifying by city, and the
  per-city train/val ratio checked, not just the global one.
- **12.2% of the subset is panoramic** (`pano_status` True) and projection types
  are mixed: perspective 1,188,459 / fisheye 633,276 / equirectangular 164,998 /
  spherical 115,248. Lima is 81.5% panoramas and Athens 68%. Feeding 360°
  equirectangular images and perspective images to the same encoder without
  handling this would let the model separate cities on projection artefacts
  rather than streetscape content — which is precisely the failure Stage 9 is
  meant to catch. Recommend filtering to perspective+fisheye, or at minimum
  recording projection as a covariate.
- Derived attributes over the subset: `green_view_index` mean 0.119 (median
  0.062), `building_view_index` mean 0.226 (median 0.191). `type_highway` is
  drive 1,567,757 / walk 383,772 / cycle 33,529 / walk-cycle 14,303 / others 27.

---

## Stage 2 — imagery ❌ BLOCKED (acceptance cannot be met)

**Stage 2 requires ~5,000 images per city. It is not possible for 9 of the 20
cities, and completely impossible for 3, because 83.8% of the subset comes from
Mapillary and Mapillary imagery requires an API access token that this machine
does not have.**

### Why

The dataset's tabular rows are pointers, not pixels. Each row carries `source`
and `orig_id`, and the image must be fetched from whichever platform published
it. The two platforms differ fundamentally:

- **KartaView** — public JSON API (`api.kartaview.org/2.0/photo/?id=…`), no
  credentials. Resolve `orig_id` → `fileurlProc` → GET the JPEG. **Works.**
- **Mapillary** — Graph API requiring OAuth. The upstream project's own
  `code/download_imgs/download_jpegs_mapillary.py` hardcodes
  `access_token = 'INSERT-YOUR-TOKEN-HERE'`, confirming a token is mandatory.

Verified directly rather than assumed:

```
$ curl "https://graph.mapillary.com/<id>?fields=thumb_2048_url"
{"error":{"message":"Invalid OAuth 2.0 Access Token","type":"MLYApiException","code":190}}
```

No Mapillary credential exists in the environment, and I did not create an
account, since registering with a third party on your behalf is your call.

### The downloader works — this is purely a credential gap

`src/imagery.py` is written, tested and ready. Measured against real rows from
the subset:

| test | result |
|---|---|
| 60 KartaView images (Singapore) | **60/60 ok**, 10.9 s at 16 workers (5.5 img/s) |
| 20 Mapillary images (Lima) | **0/20**, `error:no_mapillary_token` |
| re-run the same 60 KartaView | **60 skipped, 0.0 s** — idempotency confirmed |

Files land pre-resized to 256 px on the shortest side (measured mean 26.7 KB),
written to a `.tmp` name and atomically renamed, so an interrupted run never
leaves a partial file that a resume would mistake for complete.

### Exact scope of the shortfall

| city | KartaView available | obtainable | shortfall vs 5,000 |
|---|---:|---:|---:|
| Kuwait City | 0 | 0 | 5,000 |
| Lima | 0 | 0 | 5,000 |
| Dar es Salaam | 0 | 0 | 5,000 |
| Lisbon | 4 | 4 | 4,996 |
| Taipei | 12 | 12 | 4,988 |
| Athens | 114 | 114 | 4,886 |
| Kampala | 425 | 425 | 4,575 |
| Casablanca | 487 | 487 | 4,513 |
| Tokyo | 2,882 | 2,882 | 2,118 |
| *(the other 11 cities)* | 5,138 – 85,303 | 5,000 each | 0 |

**11 of 20 cities reach 5,000. 3 get nothing at all. Total obtainable: 58,924
of the 100,000 required.**

### What I did not do

Per the working-style rule, I did not work around this. Specifically I did not:

- silently re-pick the city list to the 11 KartaView-rich cities — that would
  destroy the regional spread the project depends on (it would drop Lima,
  Kuwait City, Dar es Salaam, Taipei, Lisbon and Athens, losing the desert,
  East Asian and two of four African streetscapes), and it would quietly
  redefine the deliverable to make a stage pass;
- lower the 5,000/city target;
- proceed to Stages 3–10 on a partial, 11-city, source-biased set. Source is
  confounded with city here (Jakarta 73.9% KartaView vs Moscow 2.7%), so an
  encoder trained on what is reachable would be free to separate cities on
  camera and processing-pipeline artefacts rather than streetscape content.

### To unblock

1. Get a free Mapillary token: mapillary.com → Dashboard → Developers → register
   an application → copy the client token.
2. `export MAPILLARY_TOKEN='MLY|...'`
3. Re-run Stage 2. `src/imagery.py` picks the token up from that variable
   automatically; no code change needed.

Projected once unblocked, from the measured rate and file size:
**100,000 images, ≈2.7 GB, ≈2.5 h at 32 workers** (≈5 h at the 16 workers
tested). Disk is not a constraint — 393 GB free.

Two alternatives, if a token is genuinely unavailable, both of which are your
call and not mine to make:

- **Re-scope to a KartaView-only city list.** Viable but weaker: it caps the
  project at roughly 11 cities and loses the desert and East Asian regions.
- **Keep 20 cities and drop the per-city target to ~2,000**, accepting that 8
  cities still fall short and 3 remain empty. This does not actually rescue it.
