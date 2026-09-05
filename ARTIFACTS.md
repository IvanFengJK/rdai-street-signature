# Reproduction manifest — artifacts not stored in git

Four binary artifacts are too large for the repository (**1.28 GB total**). The
notebook fetches them automatically on its default path; this file records what
they are, how to verify them, and how to regenerate each from scratch.

## Manifest

| File | Size | SHA-256 | Required for default notebook path |
|---|---:|---|:---:|
| `checkpoints/runA_best.pt` | 94,528,075 B (90 MiB) | `8656994a6e715947a59c378f38d9d59e7c5fb88e0297eb3c4cc9d4603a62c0a6` | **yes** |
| `data/interim/emb_trained.npz` | 456,236,109 B (435 MiB) | `f4b78ce8f24a510c95e1b56fbd4c9fa424aff4c3fad93fff4dcabd9d4595e540` | **yes** |
| `data/interim/emb_imagenet.npz` | 341,480,561 B (326 MiB) | `c9ace847611df97647d33c45ae58f80f823baf1c9dc30a475bfc54e0f81152c4` | **yes** |
| `data/interim/emb_s11_vicreg_a_ep100.npz` | 389,033,061 B (371 MiB) | `5f6c1ac09f0d1e178df12381dfcf10f823be2ffe1812ae8720741e01d2f0e201` | **yes** |

Verify after download:

```bash
sha256sum -c artifacts.sha256
```

## What each one is

### `checkpoints/runA_best.pt` — the assessed model

The trained city encoder: timm ResNet50, replaced 20-way head, trained from
random initialisation. Best epoch (19) of a 20-epoch run, validation accuracy
0.9089 on the sequence-disjoint split. Contains `model` state dict, `classes`,
the full `cfg` used, and `val_acc`.

**This is the model the project is assessed on.** Everything else is derived
from it.

*Regenerate* (~75 min on an RTX 3080 Ti; longer on a T4):
```bash
python scripts/run_stage5.py runA
```
Deterministic — `runA` and `runB` from the same seed agree bit-for-bit on every
epoch, the confusion matrix and every weight tensor.

### `data/interim/emb_trained.npz` — city-encoder embeddings

2048-d pooled backbone features for all 100,000 images, head bypassed. Arrays:
`uuid` (object, aligned to `data/interim/dataset.parquet` row order) and `emb`
(float32, 100000 × 2048).

Used by: the UMAP figure, cross-city retrieval, the city probe, the
cross-campaign diagnostic, and the entire signature analysis.

*Regenerate* (~6 min unthrottled):
```bash
python scripts/run_stage6.py
```

### `data/interim/emb_imagenet.npz` — frozen ImageNet baseline

Same architecture and extraction path, but with ImageNet-pretrained weights and
no training on this data. The untrained comparison in every retrieval table.

*Regenerate*: same command as above — `run_stage6.py` writes both.

### `data/interim/emb_s11_vicreg_a_ep100.npz` — VICReg embeddings

Backbone features from the self-supervised encoder after 100 epochs (~54 GPU-h
on a power-capped 3080 Ti). Needed to reproduce the Stage 7b comparison showing
VICReg is *not* significantly better than frozen ImageNet.

*Regenerate* (expensive — this is the 54-hour run):
```bash
python scripts/run_stage11_a.py vicreg_a          # train, 100 epochs
python scripts/run_stage11_eval.py \
    --ckpt checkpoints/stage11/vicreg_a_epoch100.pt --tag vicreg_a_ep100
```

## Hosting

> **⚠️ MANUAL STEP — these files are not yet uploaded anywhere.**
> Nothing in this repository can host 1.28 GB. Before submission, upload the four
> files as **GitHub Release assets** (2 GiB per-file limit, so all four fit) and
> set the release tag below.

Recommended: create a release tagged `v1.0-artifacts` and attach all four files
with their exact filenames. Then set, in one place only —
`ARTIFACT_BASE` at the top of the notebook, and `ARTIFACT_BASE` in
`scripts/fetch_artifacts.py`:

```
https://github.com/YOUR-GITHUB-USERNAME/street-signatures/releases/download/v1.0-artifacts
```

Any stable public HTTPS directory works — a Zenodo deposit (which also mints a
DOI, preferable for a submitted project) or a university file store are equally
fine. The fetcher only needs `<base>/<filename>` to resolve without
authentication.

**Do not use a Google Drive share link.** Drive interposes a virus-scan
interstitial for files this size, so a plain `wget`/`requests` download returns
an HTML page rather than the file, and the checksum check will fail confusingly.

## Fetching

```bash
python scripts/fetch_artifacts.py            # downloads whatever is missing, verifies checksums
python scripts/fetch_artifacts.py --check    # verify only, no download
```

The notebook calls the same module, so its default path works on a fresh Colab
with no manual download step — **once `ARTIFACT_BASE` is set**.

## If you would rather not host anything

Every artifact is reproducible from the repository plus the public dataset. The
cheap ones (`emb_trained`, `emb_imagenet`) take minutes given the checkpoint.
Regenerating `runA_best.pt` takes about an hour. Only the VICReg embeddings are
genuinely expensive, and they are needed for one comparison row.

Note that regenerating requires the **imagery**, which is ~3.14 GB fetched by
`scripts/run_stage2.py` and needs a free
[Mapillary access token](https://www.mapillary.com/dashboard/developers) in
`MAPILLARY_TOKEN`. Roughly 84% of KartaView's share of that imagery is now
cold-archived or on a dead storage node, so an exact byte-for-byte
reconstruction of the original 100,000 is no longer possible — which is the main
practical reason to host the derived artifacts rather than expect regeneration.
