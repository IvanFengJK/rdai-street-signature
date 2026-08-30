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

## Stage 2 — imagery ✅ (acceptance met, after a real fight)

You supplied a Mapillary token, which unblocked the 83.8% of the subset that
Mapillary serves. Final state: **100,000 images, exactly 5,000 per city, 20 of
20 cities at target, 3.14 GB on disk** (105,969 files fetched in total; the
extra 5,969 are over-draws, trimmed out when building the balanced manifest).
Mean file 29.0 KB, pre-resized to 256 px on the shortest side.

**Wall time: 10.1 h end to end.** That is far above the ~2.5 h I projected,
for reasons worth recording.

### Panoramas are excluded

Confirmed with you, and already what the code did. `flat_only()` keeps
`perspective` and `fisheye` only; every 360-degree frame is dropped. The
manifest is asserted panorama-free at build time and again before training.

This matters because panorama share is severely confounded with city — Lima
81.5%, Athens 68.2%, Sao Paulo 0.0%. Training across mixed projections would
let the encoder separate cities on projection geometry rather than streetscape
content, which is exactly the shortcut Stage 9 exists to detect. It costs
nothing: all 20 cities still clear 5,000 flat images (smallest pool is
Casablanca at 9,089).

### Three things went wrong, all external

1. **KartaView rate-limits under concurrency.** The bulk pass at 32 workers
   lost 3,120 Jakarta images and ~9,100 overall to HTTP errors that succeed
   immediately when retried slowly. Mapillary tolerated 32 workers fine. Fixed
   by adding retry with exponential backoff to `src/imagery.py` and driving
   KartaView through a separate globally rate-limited pass.

2. **Part of KartaView's archive is cold storage and is simply gone.** Those
   URLs return `HTTP 409` with an Azure body of
   `<Error><Code>BlobArchived</Code>` — the image cannot be served to anyone,
   and no amount of retrying changes it. About 38% of the KartaView gap was
   this. 409/410 are now treated as non-retryable so they fail fast instead of
   burning the backoff budget.

3. **One storage node is down.** `storage7.openstreetcam.org` returns
   `HTTP 502` for every request. Roughly 6% of the gap.

### How the target was still met

For each city still short, `scripts/topup_stage2.py` draws replacements from
that city's *unused eligible pool* — same flat filter, same sequence-stratified
sampling — and fetches those instead. The Stage 2 target is ~5,000 images per
city; which eligible images is a sampling detail, and every city has far more
eligible images than 5,000. This is a sampling substitution, not a relaxation
of the criterion.

Replacements prefer Mapillary, because KartaView is where the archived blobs
are. **That shifts the source mix, so here it is explicitly** — final manifest,
5,000 per city:

| city | Mapillary | KartaView | sequences |
|---|---:|---:|---:|
| Athens | 4,939 | 61 | 268 |
| Berlin | 3,992 | 1,008 | 2,012 |
| Casablanca | 4,871 | 129 | 56 |
| Dar es Salaam | 5,000 | 0 | 216 |
| Jakarta | 3,897 | 1,103 | 538 |
| Kampala | 4,918 | 82 | 639 |
| Kuwait City | 5,000 | 0 | 732 |
| Lima | 5,000 | 0 | 138 |
| Lisbon | 4,999 | 1 | 562 |
| Maseru | 5,000 | 0 | 385 |
| Melbourne | 4,619 | 381 | 932 |
| Moscow | 4,965 | 35 | 2,998 |
| Ottawa | 4,732 | 268 | 613 |
| San Francisco | 4,625 | 375 | 309 |
| San Jose | 4,675 | 325 | 789 |
| Sao Paulo | 4,704 | 296 | 2,239 |
| Singapore | 4,661 | 339 | 203 |
| Taipei | 4,994 | 6 | 989 |
| Tokyo | 4,824 | 176 | 753 |
| Washington | 4,829 | 171 | 2,138 |

Jakarta was 73.9% KartaView in the raw tabular data and is 22.1% here; Berlin
was 29.8% and is 20.2%. **The dataset is now Mapillary-dominated in every
city** (min 77.9%), which is worth knowing but is the *less* confounded
outcome: source is now much closer to uniform across cities than it was in the
raw data, so the encoder has less opportunity to separate cities on
camera/pipeline artefacts. Stage 9 should still be read with this in mind.

### Idempotency

Verified: re-running skips everything already on disk (`{'skipped': 60,
'elapsed_s': 0.0}` on a repeat of the same 60 images). Files are written to a
`.tmp` name and atomically renamed, so an interrupted run never leaves a
partial file that a resume would mistake for complete. The download was in fact
interrupted and resumed several times during those 10 hours.

**Balanced manifest: `data/interim/dataset.parquet`** — 100,000 rows, exactly
5,000 per city, zero panoramas, every row backed by a file on disk.
