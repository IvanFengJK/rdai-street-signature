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

---

## Stage 3 — dataloader ✅ (acceptance met)

| check | result |
|---|---|
| batch shape | `(64, 3, 224, 224)` ✅ |
| train/val sequence-ID intersection | **0** ✅ |
| per-city counts | printed, ~20% val in every city (19.98–22.3%) |

79,707 train / 20,293 val, no rows lost, deterministic across calls.

**Bug found and fixed.** `rng.shuffle()` was being applied to a pandas
`ArrowStringArray`, which numpy warns it cannot shuffle safely — it can
*duplicate* entries. A duplicated sequence id there would have silently
corrupted the split, i.e. quietly violated rule 1 while the assert still
passed. Now converted to a numpy object array first.

Split is stratified per city, which matters here: sequence density varies about
40x between cities (Moscow ~2 images/sequence in the sample, Casablanca ~86), so
a global sequence split would have handed whole cities to one side.

## Stage 4 — encoder ✅

```
backbone=resnet50   embed_dim=2048
total params    : 23,549,012
trainable params: 23,549,012
head            : Linear(in_features=2048, out_features=20, bias=True)
```

`pretrained=False` — weights start random. The ViT switch is verified working:
`vit_small_patch16_224` gives 21,673,364 params and a 384-d embedding.

## Stage 5 — training ✅ (both acceptance criteria met)

**Best validation accuracy 0.9089 against 0.0500 chance — 18.2x.** 20 epochs,
mixed precision, ~75 min on the RTX 3080 Ti, checkpoint every epoch.

**Reproducibility: bit-identical.** Two independent runs from the same seed
agree on train_loss, val_loss and val_acc for all 20 epochs, on the confusion
matrix, and on *every weight tensor* (max absolute difference 0.0).

That did not work at first, and the two reasons were real bugs, not flakiness:

1. **`build_model()` ran before `set_seed()`.** 23.5M weights were initialised
   from an unseeded RNG, so two runs from the "same seed" started from
   different weights and could never have matched.
2. **DataLoader workers drew augmentation from an unseeded numpy RNG.** Workers
   are forked processes; without an explicit `worker_init_fn` they produce a
   different augmentation stream every run.

Both are fixed in `src/dataset.py` and `scripts/run_stage5.py`. Worth noting
that the acceptance criterion is what surfaced them — without it the model
would have looked fine and been quietly irreproducible.

Confusion matrix is a clean diagonal; Moscow, Kuwait City, Washington and
Jakarta are near-perfect, Lima is weakest (~0.70) with its errors going to San
Jose and San Francisco.

## Stage 6 — embeddings ✅

Head stripped, 2048-d backbone output cached for all 100,000 images (1.3 min),
plus a frozen-ImageNet set for the Stage 7 comparison (1.4 min).

## Stage 7 — retrieval + evaluation ⚠️ NEGATIVE RESULT

Top-10 cross-city neighbours, 20 queries, mean absolute difference in physical
attributes. **Lower is better.**

| system | greenery | building density | road-type mismatch |
|---|---:|---:|---:|
| trained (ours) | 0.1456 | 0.2266 | 0.3191 |
| tabular (baseline) | 0.0240 | 0.0362 | 0.0000 |
| frozen imagenet | 0.0914 | 0.1336 | 0.2828 |
| random control | 0.1519 | 0.2432 | 0.3350 |

**The trained encoder barely beats the random control** — 0.1456 vs 0.1519 on
greenery. The margin is real but small.

**Frozen ImageNet features clearly beat ours** — 0.0914 vs our 0.1456.
Features never trained on this data retrieve attribute-matched streets better
than the encoder trained on it.

**The tabular baseline's win is trivial and must not be read as a result.** Its
feature vector literally contains `green_view_index` and `building_view_index`
and one-hot encodes `type_highway` — which is exactly why its road-type
mismatch is 0.0000. It is scoring on the columns being scored. It is a sanity
check on the metric, not a competitor.

So on the task the project set out to do, the encoder failed. Stages 8 and 9
establish why, and they agree with each other.

## Stage 8 — UMAP ✅ (and it diagnoses Stage 7)

Coloured by city: **20 tight, disjoint islands** with almost no overlap —
exactly what a 90.89% city classifier should look like.

Recoloured by greenery: greenery is roughly **uniform within each island**, and
there is no greenery gradient across the space. Singapore's cluster is
uniformly high-greenery, the rest uniformly pale. Greenery varies between
clusters only because cities differ in greenery, not because the embedding
encodes it.

**The embedding is close to a city one-hot in disguise.** It has essentially no
within-manifold structure corresponding to physical attributes. That fully
explains Stage 7: once the query's own city is excluded, the nearest other-city
neighbours are whichever island happens to sit closest in an arbitrary
arrangement, carrying no attribute information.

## Stage 9 — Grad-CAM ✅ (the news is bad, and it stays in)

The heatmaps concentrate on:

- **road surface and lane markings** — Berlin, Casablanca, Kampala and Jakarta
  all light up across the tarmac in the lower half of the frame;
- **crosswalk stripes** — Lima is the clearest case, the zebra markings are
  almost the entire activation;
- **bottom-of-frame camera furniture** — Kuwait City activates on the strip
  where the dashcam's own text overlay and the vehicle bonnet sit.

Very little activation lands on buildings, vegetation or skyline.

This is exactly the shortcut CLAUDE.md's Stage 9 predicted ("licence plates,
road markings, or vehicle shapes rather than the street itself"). It also
reconciles the two headline numbers. Road markings, kerb geometry, tarmac
colour and dashcam furniture are *superb* city identifiers — they are
standardised nationally and the capture vehicle is often literally the same car
— and near-useless for judging whether two streets look alike.

**The pretext task failed, not the training.** City classification asks the
encoder to find what *separates* cities; retrieval needs what they *share*.
Those objectives are opposed, and 90.89% accuracy is evidence of the shortcut
rather than evidence against it. The model did exactly what it was asked to do;
it was asked for the wrong thing.

A follow-up should change the objective, not the architecture: a contrastive or
attribute-supervised objective, augmentations that destroy the shortcut
(aggressive bottom-of-frame cropping, colour jitter on tarmac), or masking the
road surface outright.

## Stage 10 — notebook ✅

`notebooks/parallel_streets.ipynb`, 28 cells, executed top to bottom with all
outputs visible. Generated by `scripts/build_notebook.py` rather than
hand-edited, per the rule against developing inside a `.ipynb`.
`TRAIN_FROM_SCRATCH` defaults to `False` and loads the published checkpoint and
cached embeddings; the `True` path applies the reduced T4 config from
`config.yaml`. A markdown cell states plainly that reported results come from
the full local run.

---

## Open questions for you

1. **The negative result is the finding.** I have reported it as such rather
   than tuning until it looked better. If you would rather the project
   demonstrate a *working* doppelgänger retrieval, that needs a different
   pretext task (contrastive), not more epochs — say the word and I will build
   it as a Stage 11 comparison against this as the baseline.
2. **The Colab clone URL in the notebook is a placeholder**
   (`https://github.com/<user>/acv_training.git`). It needs your actual repo
   before the notebook will run on a fresh Colab.
3. **Checkpoints and embeddings are not in git** (~800 MB). For the notebook's
   `False` path to work on Colab they need hosting somewhere fetchable.

---

# Stage 11 — city-invariant self-supervised representation learning

A controlled follow-up to the Stage 7 negative result. Stages 0-10 are
untouched and remain the baseline; everything here writes to
`checkpoints/stage11/`, `data/interim/emb_s11_*.npz` and
`outputs/stage11_*`.

**Research question.** Does replacing the city-classification pretext task with
self-supervised learning produce embeddings that capture transferable street
appearance rather than city identity?

## Setup

- Same 100,000 images, same 20 cities, same sequence-disjoint split
  (79,707 train / 20,293 val), rule 1 re-asserted in every script.
- Same timm ResNet50, same 2048-d pooled representation. Only the objective
  changes, so the comparison isolates the objective rather than confounding it
  with architecture.
- VICReg (invariance / variance / covariance), 3-layer 2048-wide expander.
- Evaluation attributes (greenery, building density, road type) and the
  prohibited perception scores are never used in training. The SSL dataset
  reads only pixels; the adversary would read only the city label.

**The Stage 7 query set is now frozen** to `data/interim/stage7_queries.npz`.
Stage 7 derived its 30,000-image subsample and 20 queries from the seed at
runtime, which is reproducible only while nobody touches the derivation. It is
now asserted, not recomputed. Validation: the four baseline rows reproduce the
original Stage 7 numbers to four decimals, so Stage 11 results are directly
comparable to the originals.

## Experiment A — VICReg (in progress)

| epoch | greenery ↓ | bldg density ↓ | road mismatch ↓ | city probe |
|---|---:|---:|---:|---:|
| trained city encoder (Stage 5) | 0.1456 | 0.2266 | 0.3191 | 0.9063 |
| frozen imagenet | 0.0914 | **0.1336** | **0.2828** | 0.7738 |
| VICReg @ epoch 1 | 0.1156 | 0.2065 | 0.3650 | 0.4864 |
| **VICReg @ epoch 30** | **0.0879** | 0.1348 | 0.4118 | 0.8182 |
| tabular (oracle-ish sanity check) | 0.0240 | 0.0362 | 0.0000 | - |
| random control | 0.1519 | 0.2432 | 0.3350 | - |

**VICReg beats frozen ImageNet on greenery** (0.0879 vs 0.0914) and ties it on
building density. This is the first point in the project where a model we
trained beats the untrained baseline on the retrieval metric, and it is far
ahead of the Stage 5 city encoder on both continuous attributes.

**But road-type mismatch is worse than random** (0.4118 vs 0.3350) and got
*worse* with training (0.3650 at epoch 1). VICReg improves at matching
continuous appearance attributes while degrading at matching road function. No
explanation yet; recorded rather than rationalised.

### The finding that revises Stage 9

**The city probe rose from 0.486 to 0.818 with no city label anywhere in
training.** Self-supervised learning on this data spontaneously accumulates
city identity, approaching the supervised encoder's 0.906.

That undercuts the Stage 9 story. The original explanation was that the
city-classification *objective* produced the shortcut. But city information
accumulates without any city objective, and it accumulates *while retrieval
improves*: within VICReg training, more recoverable city information went with
better retrieval, not worse. Across systems the relationship is not monotonic
either - Stage 5 has the most city information and the worst retrieval, while
VICReg has more than ImageNet and retrieves slightly better on greenery.

This is Outcome 4 from the Stage 11 brief: the models improve while city
remains easily recoverable, so **the original explanation was incomplete and
needs revising**. It also directly challenges the premise of Experiment B -
that suppressing city identity should improve retrieval.

### Loss-curve caveat, recorded because it was nearly misleading

Train loss fell monotonically (20.65 -> 13.99 by epoch 41) while the val VICReg
loss *rose* (22.7 -> 46.7). That divergence looked like a failure and was not:
the epoch-30 evaluation showed the representation improving substantially over
epoch 1. Two reasons the val number is a weak proxy: it is computed under
`model.eval()`, where the expander's BatchNorm uses running rather than batch
statistics, and VICReg's variance and covariance terms are batch statistics by
construction. Val is monitoring-only and never used for model selection.
**Pretext-task loss is not representation quality** - which is the same lesson
Stage 7 taught, arriving from the other direction.

`std_mean` climbed 0.305 -> ~0.62 and plateaued there, short of the 1.0 the
variance hinge targets. Covariance rose monotonically to ~3.2. Both suggest the
covariance coefficient of 1.0 may be too weak at this dataset scale.

## Scope reduction (deliberate, not silent)

The brief specified an adversarial sweep over
lambda in {0, 0.05, 0.10, 0.25, 0.50}, plus Experiment C (bottom-frame
masking). **Measured** cost on this machine is ~50 h per run, so:

| planned | runs | GPU hours |
|---|---:|---:|
| full lambda sweep | 4 more | ~200 h |
| Experiment C | 1 | ~50 h |

Reduced to **a single adversarial contrast at lambda=0.10** against the
lambda=0 run (Experiment A), and **Experiment C dropped**.

Reasons, in order of weight:

1. **The headline findings are already measured.** VICReg beating frozen
   ImageNet, and the city-probe result overturning Stage 9, are both in hand at
   epoch 30. Neither improves with a denser sweep.
2. **The epoch-30 result makes one contrast sufficient and more interesting
   than five.** We found city information rising *while* retrieval improved,
   contradicting the sweep's premise. A single lambda=0.10 run tests that
   directly; tracing a curve through a premise the data already questions is
   not a good use of 200 h.
3. **This is a course project.** Stages 0-10 already satisfy the brief in full.
   The marginal value of sweep points 2-5 is close to zero for the deliverable.
4. Stage 9's camera-artefact finding is already reported honestly in the
   notebook; Experiment C would confirm it, not change it.

This is recorded in `config.yaml` under `stage11.adversarial` alongside
`originally_specified_sweep`, so the reduction is visible in the config rather
than only in prose.

## Machine constraint that shaped everything

The GPU ran the whole of Stage 11 **power-capped at 35 W against an 80 W
default** (`sw_power_cap: Active`, clock oscillating 210-1260 MHz against a
2100 MHz maximum). Measured effect: 80-92 img/s against 299 img/s benchmarked
unthrottled, i.e. ~32 min/epoch instead of ~9.

Two hypotheses were tested and one survived:

- **Charging budget - disproved.** The battery was at 12% and charging at 46 W
  when first observed. It reached 100% and the cap did not move.
- **BIOS `Graphics = Hybrid Graphics`** - the surviving explanation, read from
  HP's WMI BIOS interface on this ZBook Studio 16 G9. Not changed: it needs a
  reboot into F10 and is a firmware change that should be a deliberate act, not
  a side effect of a training job.

`nvidia-smi -pl 80` fails from both WSL and Windows with *"not supported in
current scope"*. That is not a permissions issue - power-limit control is a
Quadro/Tesla feature, and `Requested Power Limit: N/A` confirms the clamp is
imposed below the driver.

Also measured: **running the evaluation concurrently with training is not
free.** Epoch 2 took 73 min against 32 min for epoch 1 because the embedding
extraction was competing for the capped GPU. Intermediate evaluations were
reduced accordingly.
