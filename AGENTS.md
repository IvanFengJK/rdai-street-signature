# Parallel Streets — Street Doppelgänger Atlas

## What this is

A computer vision course project. We train an image encoder on street-level
imagery, then use the learned embedding to find visually similar streets in
distant cities.

**Deliverable:** one public Google Colab notebook that runs top to bottom with
all outputs visible. Deadline 5 Oct 2026.

**The point of the project is that I trained a model.** The course covers many
pretrained models (SAM, CLIP, Florence-2, YOLO, diffusion). Running those is not
what is being assessed. The trained encoder is the deliverable; everything else
supports it.

## Hard constraints

- **PyTorch + timm.** Not JAX, not TensorFlow. No TPU code paths.
- Development on a local RTX 3080 (16GB, CUDA 13.2). Final notebook must also
  run on a Colab T4.
- **Pin dependency versions to what Colab currently ships**, not to latest.
  Check Colab's versions before pinning.
- Python modules in `src/`, driven by `config.yaml`. The notebook is written
  **last** and is a thin layer that imports from `src/` and displays results.
  Do not develop inside a `.ipynb`.

## Data

NUS Global Streetscapes — `NUS-UAL/global-streetscapes` on HuggingFace.
10M street-level images, 688 cities, crowdsourced from Mapillary and KartaView,
enriched with ~346 features.

The HuggingFace repo hosts **only tabular data**. Imagery downloads separately
via the project's GitHub wiki. Cache everything to local disk; never re-download.

Related: `github.com/ualsg/global-streetscapes`, `info.csv` documents the columns.

## Non-negotiable rules

1. **Split by sequence ID, never randomly.** Mapillary images come in sequences
   shot metres apart. A random split puts near-duplicates in both train and test
   and produces meaningless accuracy. `assert` zero sequence overlap between
   splits, and fail loudly if violated.

2. **Do not use the perception scores.** The `safety`, `wealth`, `beauty`,
   `lively`, `boring`, `depressing` columns come from crowdsourced ratings with
   documented bias problems. Labelling streets in specific countries as looking
   unsafe or poor is not something this project does. Use greenery, building
   density, road type, and similar physical attributes instead.

3. **`sklearn.NearestNeighbors` on tabular features is a baseline, not the
   method.** Retrieval runs on the trained visual embedding.

4. **Three retrieval systems, evaluated against each other:**
   - the trained encoder's embedding (ours)
   - tabular feature vector, normalised
   - frozen ImageNet features, no training
   All three answer the same query set.

5. **Evaluation metric.** For each query, take top-k neighbours *in other cities*
   and measure mean absolute difference in physical attributes versus randomly
   paired streets. A good embedding retrieves streets matching on attributes it
   was never shown. Report all three systems on this.

6. **No web UI, no interactive globe, no Gradio.** Static matplotlib map and
   paired image cards.

## Stages

Complete one stage, report results.

**Stage 0 — schema.** Fetch `info.csv`. Write `schema.json` recording the real
column names for city, sequence ID, greenery, building density, road type.
Do not guess these; I have not verified them either. Show me the list.

**Stage 1 — subset.** Pick 15–20 cities across visually distinct regions,
including Singapore. Filter tabular rows. Report per-city image counts.
Acceptance: no city under 2,000 images.

**Stage 2 — imagery.** Download ~5,000 images per city. Pre-resize to 256px on
disk. Idempotent: skip anything already fetched. Report total size and wall time.

**Stage 3 — dataloader.** Acceptance: yields batches of shape (B,3,224,224);
train/val sequence-ID intersection is empty; per-city counts printed.

**Stage 4 — encoder.** timm ResNet50, replaced head, city classification as the
pretext task. Print architecture and parameter count. Config-switchable to a
small ViT so ResNet-vs-ViT can be compared later.

**Stage 5 — train.** Mixed precision, checkpoint every epoch. Report loss curve
and confusion matrix. Acceptance: val accuracy clearly above chance, and
training a second time from the same seed reproduces it.

**Stage 6 — embeddings.** Strip the head, extract and cache embeddings for every
image.

**Stage 7 — retrieval + evaluation.** All three systems, the metric from rule 5,
same 20 queries. One results table.

**Stage 8 — UMAP.** Coloured by city, then recoloured by greenery.

**Stage 9 — Grad-CAM.** The model may be reading licence plates, road markings,
or vehicle shapes rather than the street itself. Find out and report it plainly.
This section stays in the notebook even if the news is bad.

**Stage 10 — notebook.** Now, and not before. Thin layer over `src/`.
`TRAIN_FROM_SCRATCH` flag defaulting to `False`, loading published checkpoint
and cached embeddings. The `True` path must genuinely run on a T4 with a
reduced config. A markdown cell states plainly that reported results come from
the full local run.

## Working style

- Small commits, one per stage.
- When something fails, say so and stop. Do not silently substitute a smaller
  model, a random split, or synthetic data to make a stage pass.
- If an acceptance criterion cannot be met, tell me why rather than working
  around it.
- Flag any assumption you are making about data structure before building on it.