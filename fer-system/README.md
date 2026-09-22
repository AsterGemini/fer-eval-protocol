# fer-system

Runnable companion to the **evaluation / data-merge protocol**.

**Read first:** [PROTOCOL.md](PROTOCOL.md) — frozen official FER val, no
circular val overlay, RAF-DB as selection metric (`raf_db.macro_f1`),
near-dup filtering when merging Kaggle packs, hashed mix + public recipe.

The trainer is a standard EfficientNet (timm) fine-tune. That is intentional:
this repo’s value is **how we measure and merge data**, not a novel backbone.
Do not treat Gradio or reported FER-val accuracy as a SOTA claim.

Default train / eval / demo config: `configs/public.yaml`.
Private ExpW packs belong in a gitignored `configs/private.yaml` — never copy
them into `public.yaml`. No faces or `.pt` weights ship in git.

## Overview

- **Task**: Classify facial expressions into 7 emotion classes
- **Training data**: Primary FER2013-style split + extra train sources (see [DATA_SOURCES.md](DATA_SOURCES.md))
- **FER val**: Official labels, frozen (`jonathanoheix/.../images/validation`)
- **Checkpoint metric**: `raf_db.macro_f1` (eval-only RAF-DB val)
- **Backbone**: EfficientNet-B0 (timm), swappable via `--model b0|b1|b2`
- **Training**: Full-model fine-tune in `public.yaml`; `--full-backbone` is explicit; last-4 remains a supported yaml profile

Classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

### Training sources (summary)

Public-slug recipe (`configs/public.yaml`, documented mix **v1.4.0**) is
the default train path. Extra minority packs in v1.3.0–v1.5.0 stay
preparable; `kdef` and `mma_minority` are excluded from this default.

| Role | Source | Kaggle |
|------|--------|--------|
| Train + val (primary) | Cleaned FER2013-style | `jonathanoheix/face-expression-recognition-dataset` |
| Extra train + eval-only val | RAF-DB basic 7 emotions | `nishchalchandel/raf-db-face-emotion-dataset` |
| Train only (extra) | AffectNet 7-class subset | `mstjebashazida/affectnet` |
| Train only (extra) | ExpW in-the-wild faces (raw pack → crops) | `shahzadabbas/expression-in-the-wild-expw-dataset` |
| Train only (extra) | FER+ high-agreement disgust/fear | `deadskull7/fer2013` + Microsoft FER+ votes |
| Train only (extra) | CK+48 lab-posed disgust/fear | `shawon10/ckplus` |
| Train only (extra) | AffectNet-family disgust/fear (near-dup-filtered) | `prasadsomvanshih/fer2013-affectnet-dataset-emotions` |

**Not in the default `public.yaml` mix (optional / preparable only):**

| Role | Source | Kaggle |
|------|--------|--------|
| Optional train only | KDEF lab-posed disgust/fear | `chenrich/kdef-database` |
| Optional train only | MMAFEDB train disgust/fear (near-dup-filtered) | `mahmoudima/mma-facial-expression` |


## Acknowledgments

This pipeline builds on publicly released facial-expression datasets and
redistributions. We thank the original authors and maintainers. **Dataset
licenses are separate from this repo’s MIT code license** — check each source
before redistribution. Faces and trained weights are not shipped here.

| Source (role in mix v1.4.0) | Original work / note | Kaggle pack used here |
|----------------------------|----------------------|------------------------|
| FER2013-style ImageFolder (primary train + frozen val) | Goodfellow et al., *Challenges in Representation Learning* (ICML 2013 workshop); cleaned ImageFolder packaging | [`jonathanoheix/face-expression-recognition-dataset`](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) |
| RAF-DB (extra train + selection val) | Li, Deng, Du, *Reliable Crowdsourcing and Deep Locality-Preserving Learning…* (CVPR 2017 / TIP) | [`nishchalchandel/raf-db-face-emotion-dataset`](https://www.kaggle.com/datasets/nishchalchandel/raf-db-face-emotion-dataset) |
| AffectNet 7-class subset (extra train) | Mollahosseini, Hasani, Mahoor, *AffectNet* (IEEE TAC 2017) | [`mstjebashazida/affectnet`](https://www.kaggle.com/datasets/mstjebashazida/affectnet) |
| ExpW (extra train crops) | Zhang et al., Expression in-the-Wild; raw pack → box crops | [`shahzadabbas/expression-in-the-wild-expw-dataset`](https://www.kaggle.com/datasets/shahzadabbas/expression-in-the-wild-expw-dataset) |
| FER+ high-agreement disgust/fear (extra train) | Barsoum et al., *Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution* (ICMI 2016); votes over FER2013 | [`deadskull7/fer2013`](https://www.kaggle.com/datasets/deadskull7/fer2013) + FER+ labels |
| CK+48 (extra train) | Lucey et al., *Extended Cohn-Kanade (CK+) Dataset* (CVPRW 2010) | [`shawon10/ckplus`](https://www.kaggle.com/datasets/shawon10/ckplus) |
| Minority-boost AffectNet-family pack (extra train, near-dup filtered) | AffectNet-family Kaggle remix; we drop near-duplicates of FER/RAF/AffectNet | [`prasadsomvanshih/fer2013-affectnet-dataset-emotions`](https://www.kaggle.com/datasets/prasadsomvanshih/fer2013-affectnet-dataset-emotions) |

Optional preparable packs (not in the default v1.4.0 recipe): **KDEF** (Lundqvist et al.) via [`chenrich/kdef-database`](https://www.kaggle.com/datasets/chenrich/kdef-database); **MMAFEDB** train minority via [`mahmoudima/mma-facial-expression`](https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression).

Backbone / tooling: [timm](https://github.com/huggingface/pytorch-image-models) EfficientNet, PyTorch, Gradio. Eval-protocol ideas were stress-tested against common FER2013 transfer practice (official-label val noise ceiling ~70%).

## Project Structure

```text
fer-system/
├── configs/           # YAML hyperparameters (`public.yaml` is the default)
├── data/              # Download + transforms + manifest.json
├── src/               # Dataset, model, train, evaluate, baseline, inference
├── app/               # Local Gradio demo (inference + Haar face crop)
├── notebooks/         # Exploration only
├── scripts/           # train.sh, ab_run.sh (A/B harness)
├── artifacts/         # Per-model checkpoints, logs, figures (gitignored; .gitkeep only)
├── artifacts_from_googlecolab/  # Canonical Colab export (gitignored)
├── PROTOCOL.md        # Evaluation / data-merge protocol (the contribution)
├── VERIFY.md          # Pre-publish smoke checks (no full train claimed)
├── DATA_SOURCES.md    # Which datasets train vs validate
├── requirements.txt
├── requirements-app.txt  # Gradio + OpenCV (local demo only)
└── README.md
```

## Setup

```bash
cd fer-system
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Local demo (bring-your-own checkpoint). From `fer-system/`:

```bash
pip install -r requirements-app.txt
python app/app.py --checkpoint /path/to/local.pt
```

Default `python app/app.py` loads
`artifacts/models/efficientnet_b0/checkpoints/best.pt` after you train
(`python -m src.train --model b0`). Missing weights exit with a hint — this
repo does not ship trained models and does not publish a weight URL.
`--ensemble` needs both yaml members on disk. Upload a photo; Haar crops
the largest face. No face → a warning, not a guess. Research visualization,
not a production emotion detector.

Train later. Rented GPU (Vast.ai): use your own GPU host checklist (Vast playbooks stay in the private maintainer clone).

Datasets resolve via kagglehub on first train, including ExpW extract/crop
into the local cache. On Colab choose B0 or B2 directly (`p7zip-full`
installs itself if missing):

```bash
# From fer-system/ — public-slug recipe (`configs/public.yaml`)
bash scripts/train.sh --model b0
bash scripts/train.sh --model b1
bash scripts/train.sh --model b2 --epochs 200 --patience 16
```

The direct CLI equivalents are:

```bash
python -m src.train --model b0
python -m src.train --model b1
python -m src.train --model b2 --epochs 200 --patience 16
python -m src.train --model b1 --dry-run
```

B0, B1, and B2 each store independent state under
`artifacts/models/<backbone>/`. B1 and B2 start from ImageNet
weights on their first run; later runs resume only that model's checkpoint.
Their in-run Pareto veto is
disabled so a candidate checkpoint is available for post-training comparison
without risking the held B0 checkpoint. All models stay at the config's
224×224 so they can share one inference transform in an ensemble.
Use `--artifacts-root PATH` to isolate another run explicitly.

Optional: refresh `data/manifest.json` fingerprints without training:

```bash
python -m data.download --config configs/public.yaml
```

Details: [DATA_SOURCES.md](DATA_SOURCES.md).

After training, check `artifacts/models/efficientnet_b0/figures/` for `loss_curves.png`, `update_ratio.png`,
`class_loss_curves.png`, and `metrics.json` (includes hyperparameters + history).

Evaluate a checkpoint:

```bash
python -m src.evaluate --config configs/public.yaml --checkpoint artifacts/models/efficientnet_b0/checkpoints/best.pt
```

Evaluate B0 + B2 with equal-weight probability averaging before enabling it:

```bash
python -m src.evaluate \
  --config configs/public.yaml \
  --checkpoint artifacts/models/efficientnet_b0/checkpoints/best.pt \
  --checkpoint artifacts/models/efficientnet_b2/checkpoints/best.pt \
  --output-dir artifacts/models/ensemble_b0_b2/eval
```

That always-on run is the latency/accuracy baseline (`deferral_rate=1.0`). Entropy cascade runs B0 first and calls B2 only when B0’s predictive entropy (nats) exceeds `max_entropy`:

```bash
python -m src.evaluate \
  --config configs/public.yaml \
  --checkpoint artifacts/models/efficientnet_b0/checkpoints/best.pt \
  --checkpoint artifacts/models/efficientnet_b2/checkpoints/best.pt \
  --weight 0.8 --weight 0.2 \
  --cascade --max-entropy 0.8 \
  --output-dir artifacts/models/ensemble_b0_b2/cascade_e08
```

`max_entropy: 0.8` in `configs/public.yaml` is a starting knob (`log(7) ≈ 1.95` is uniform). It is **not** tuned: re-run with `0.4 / 0.8 / 1.2` on RAF-DB val and keep the highest threshold that still beats held B0 macro-F1 (81.71%). Metrics JSON includes `deferral_rate` (fraction of images that ran B2).

Only enable `inference.ensemble.enabled` in `configs/public.yaml` if the ensemble
beats the best individual checkpoint on the frozen validation metric. Repeat
`--weight` once per checkpoint to test non-equal positive weights. Cascade
settings apply when two members are loaded; they do not turn the demo into an
ensemble by themselves.

Run inference on a single image (expects a cropped face):

```bash
python -m src.inference \
  --config configs/public.yaml \
  --checkpoint artifacts/models/efficientnet_b0/checkpoints/best.pt \
  --image path/to/face.jpg
```

Repeat `--checkpoint` to use an ensemble:

```bash
python -m src.inference \
  --config configs/public.yaml \
  --checkpoint artifacts/models/efficientnet_b0/checkpoints/best.pt \
  --checkpoint artifacts/models/efficientnet_b2/checkpoints/best.pt \
  --image path/to/face.jpg
```

Add `--cascade --max-entropy 0.8` to skip B2 on low-entropy (confident) faces.
`--json` then includes `cascade.entropy` and `cascade.deferred`.

### Local demo (Gradio)

Default `python app/app.py` loads yaml `paths.best_checkpoint` (B0) when that
file is on disk; otherwise it exits and tells you to train or pass a local
`.pt`. Point `--checkpoint` or `FER_CHECKPOINT` at a file you bring (repeat
`--checkpoint`, or use `FER_CHECKPOINTS`, for an ensemble). Pass `--ensemble`
to load yaml `inference.ensemble.members` (80/20 B0/B2) without editing
`enabled` — both files must exist. `FER_CHECKPOINTS` as a comma-separated
override, or the legacy `FER_CHECKPOINT` single-model override, still win
over CLI and yaml. Optional `FER_ENSEMBLE_WEIGHTS` is also comma-separated.
When two members are loaded, nested `inference.ensemble.cascade` (or
`FER_CASCADE_MAX_ENTROPY`) can skip B2 on low-entropy faces.
Weights are not shipped. The B0 RAF-DB 81.71% figure applies only when that
B0 checkpoint is loaded.

```bash
pip install -r requirements-app.txt
python app/app.py --checkpoint /path/to/local.pt
python app/app.py
python app/app.py --ensemble
```

Photos with no detectable face return a warning instead of a guess.

A/B two configs (fresh seed each; appends `artifacts/experiments.csv`):

```bash
bash scripts/ab_run.sh configs/public.yaml path/to/other.yaml
```

Compare existing `run_*.json` logs (flat payload; no nested history walk):

```bash
python -m src.compare_runs --dir artifacts/models/efficientnet_b0/logs
python -m src.compare_runs --dir artifacts/models/efficientnet_b0/logs --format text
```

## Baselines

Run trivial baselines on the val split before comparing models:

```bash
python -m src.baseline --config configs/public.yaml
```

| Baseline | Accuracy | Macro-F1 | Notes |
|----------|----------|----------|-------|
| Majority class | 0.258 | 0.059 | Always predicts `happy` |
| Random chance | 0.143 | 0.133 | Uniform over 7 classes |
| Published FER-2013 SOTA | ~0.70–0.73 | — | Transfer + ensembles |

Results are written to `artifacts/models/efficientnet_b0/figures/baselines.json`. Current best
(EfficientNet-B0 + RAF-DB): see `artifacts/models/efficientnet_b0/figures/metrics.json`.
Checkpoints are not shipped; train or point `--checkpoint` at a local `.pt`.

Data versioning: `python -m data.download` writes `data/manifest.json`
(counts + SHA256 fingerprints) stamped **v1.4.0**, matching the public mix
(FER2013 + RAF-DB + AffectNet + public ExpW + FER+/CK+/minority_boost).
The committed SHA snapshot is that v1.4.0 fingerprint.
v1.3.0–v1.5.0 minority sources remain preparable for isolated ablations;
`kdef` and `mma_minority` stay out of the default recipe. Training run
reports include `data_version` and `data_fingerprint`.

## Design Notes

| Choice | Rationale |
|--------|-----------|
| Config-driven hyperparameters | Reproducibility; no magic numbers in training code |
| Class-weighted CE only | Single imbalance correction; avoids rare-class memorization from WeightedRandomSampler+replacement |
| Macro F1 as checkpoint metric | Accuracy is misleading under imbalance; monitor is `raf_db.macro_f1`, not FER val |
| Freeze profile | Public default is full-model; `--full-backbone` is explicit; last-4 remains a supported yaml profile |
| `resume_from_best: true` default | Continuous fine-tuning from `best.pt` when present |
| Grayscale → 3-channel 224×224 | Match ImageNet pretrained backbone expectations |
| Flip / rotation / affine / jitter / erase | Milder face-safe aug; hue/sat jitter are no-ops after Grayscale(3) |
| Cosine schedule + `eta_min_ratio: 0.1` | Floor = 10% of the yaml base LR |
| Swappable timm backbone | Iterate architectures without rewriting the pipeline |
| Loss + update/weight plots | Track learning health (Karpathy ~1e-3); `update_ratio_every: 0` for metric runs |
| Per-class val loss | Spot hard emotions under imbalance |
| Hyperparameters in metrics.json | Experiment tracking without digging through checkpoints |
| `src/compare_runs.py` | Flat compare of `run_*.json` (plateau / abort / Pareto) for agents |
| `scripts/ab_run.sh` | Automates the METRIC check (`raf_db.macro_f1`) across configs |
| `src/baseline.py` + `data/manifest.json` | Hard floor + data versioned before any tuning |
| Entropy cascade (opt-in) | B0 always; B2 only when B0 entropy exceeds `max_entropy` (uncalibrated until RAF sweep) |

## Loop changelog (5-step grounded improvement)

Early FER-val loop (historical; later work moved the *selection* metric to
RAF-DB — see [PROTOCOL.md](PROTOCOL.md)). Pre-loop FER-val macro F1
**0.3400** → best measured **0.3902**.

| Step | Change | Evidence |
|------|--------|----------|
| Questioned | Ownerless disgust override, resume defaults, hue/sat | Config/README audit |
| Deleted | Hue/sat jitter | Provable no-op after Grayscale(3) |
| Deleted | Class-weighted CE + `disgust: 2.8` override | Sampler-only → F1 **0.3648** (Pass 1) |
| Simplified | Stock cosine; val-only eval loader | Less schedule/IO complexity |
| Accelerated | AMP + shorter fine-tune verify budget | ~22 min Pass-2 on MPS |
| Automated | `scripts/ab_run.sh` | METRIC gate across configs |
| Corrected | `unfreeze_last_n_blocks: 4` (not 2) | last_n≤2 adds 0 backbone params on EfficientNet-B0 |
| Kept (Pass 2) | Partial backbone fine-tune | F1 **0.3648 → 0.3902** |

**Pass 3**: STOPPED — F1 0.390 still below 0.45; further pipeline churn unlikely to close the gap (FER2013 label noise).

## Known limitations

- FER val is label-noise-capped (~0.70–0.73 published transfer band). The early
  FER-val F1 **0.3902** loop stopped for that reason; later runs select on
  RAF-DB, not by chasing FER val.
- The 0.45 / 0.50 FER-val targets were estimates from published FER2013
  transfer results, not guarantees on this split.
- No held-out test set beyond eval-only RAF-DB val; FER val is a frozen
  report split, not a tuning signal.
- `happy` anomaly could also have been a class-mapping bug; confusion-matrix
  inspection still recommended when F1 stalls.

## Roadmap (later)

- [x] Face detection preprocessing for in-the-wild images (Haar, in `app/`)
- [ ] Real-time webcam inference
- [x] Gradio local demo (`app/`)
- [ ] ONNX export for deployment (weights not redistributed from this dump)

## License

Code is [MIT](../LICENSE). Datasets are third-party (Kaggle / original
authors) and are **not** covered by that license — download them yourself
and read each source's terms before any use or redistribution. Face images
and trained weights (`.pt` / `.pth` / `.onnx`) are gitignored and not part
of this dump. AffectNet / RAF-DB / ExpW terms often forbid redistributing
trained models. There is no hosted weight URL; train locally or bring a
checkpoint.
