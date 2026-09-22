# Evaluation and data-merge protocol

**This document is the contribution of the public dump.** The rest of
`fer-system/` is a runnable companion (EfficientNet fine-tune + Gradio).
Architecture is commodity; these measurement rules are what to cite or
compare against. The trainer is a standard EfficientNet-B0 fine-tune; what
is not standard is how validation, checkpointing, and extra Kaggle packs
are treated.

Public-slug recipe: [`configs/public.yaml`](configs/public.yaml) (default
train path). Dataset roles: [`DATA_SOURCES.md`](DATA_SOURCES.md). Private
ExpW packs belong in a gitignored local file such as `configs/private.yaml`
— do not copy them into `public.yaml`.

## Frozen official FER val

Primary validation is the official FER2013-style val split
(`jonathanoheix/.../images/validation`). Files and **labels** stay frozen.

FER2013 single labels are noisy. Published transfer numbers on this split
often sit around 70–73% accuracy; treating that as a headroom target for a
single model is a mistake. Once the val curve plateaus there, further
pipeline churn is usually fitting noise.

All FER val numbers in this project are official-label numbers, comparable
to the literature. They are **not** the checkpoint metric.

## Why the val overlay was circular

A 25-error human audit (`data/label_review/`) marked official labels
Correct / Wrong / Unsure. An overlay then remapped 13 “wrong” labels to the
**audit-time model prediction**. Humans never named a replacement class.
The same model was confidently wrong on 8 of 21 decided cases.

That is circular evaluation: the selection signal is rewritten toward the
model being selected. The overlay was reverted on 2026-08-24
(`data.val_label_corrections` must stay `null`). Audit JSON stays as
documentation only. Crops from that review are not training data.

## RAF-DB as the selection metric

Checkpointing and early stopping monitor `raf_db.macro_f1` (eval-only RAF-DB
val). RAF-DB is a different real-world collection with cleaner labels and
more dynamic range than FER val.

FER val is still computed and reported. It is a regression floor and a
literature-comparable number, not the thing that writes `best.pt`.

Merging another FER2013 redistributed pack into train is not “more data.”
It duplicates the primary set and risks train/val leakage. RAF-DB is in the
mix because it is a different collection with the same seven classes.

## Near-dup filtering when merging Kaggle packs

Kaggle emotion packs often remix FER2013. The pack
`prasadsomvanshih/fer2013-affectnet-dataset-emotions` advertises that in
its name. Perceptual hash (16×16 aHash, Hamming ≤ 10 on 48×48 grayscale)
found ~895 fearful and ~97 disgusted crops that are near-duplicates of
**primary FER val**, plus overlap with FER train and the existing AffectNet
extra source.

`data/prepare_minority_boost.py` keeps a disgust/fear crop only when its
hash is > 10 bits from every image in:

- primary FER train **and** val (hard fail if FER val cannot be resolved)
- existing AffectNet train
- RAF-DB train and val (guard; measured 0 hits)

Recompressed duplicates sit at distance 0–1; different faces cluster near
~128. Adding that pack raw would leak the frozen val split into training.

FER+ minority rows are Training-only (PublicTest/PrivateTest never enter).
CK+48 contributes lab-posed disgust/fear only. Neither is a second val
split.

## Failed levers that were deleted

These were tried, measured, and must not be reintroduced as defaults:

| Deleted | Why |
|---------|-----|
| Hue/saturation jitter | No-op after `Grayscale(3)` |
| `WeightedRandomSampler` stacked with class-weighted CE | Rare-class memorization; one imbalance lever only |
| Absolute class-weight overrides (`disgust: 2.8`, fear 1.5/2.0) | Fear val CE climbed (2.9→4.1) with F1 stuck ~0.58 |
| Fear `class_weight_multipliers: 1.15` | Did not beat the RAF F1 of sad-only 1.5× |
| Val-label overlay toward the model prediction | Circular eval (above) |
| Constant LR (`eta_min_ratio: 1.0`) | Floor = base LR; cosine was a no-op |
| Restarting a resume at 4e-4 after a ~4e-5 run | RAF F1 dropped and did not recover |

What survived: auto inverse-frequency class weights, optional **sad 1.5×**
multiplier, label smoothing 0.1 (train only), cosine with
`eta_min_ratio: 0.1`, full-model fine-tune, RAF Pareto class-F1 gate for
weak classes.

## What this dump does not include

- Face images, trained checkpoints, or ONNX weights (gitignored). AffectNet /
  RAF-DB / ExpW terms often forbid redistributing trained models. Train
  locally (`python -m src.train --model b0`, which writes
  `artifacts/models/efficientnet_b0/checkpoints/best.pt`) or point
  `--checkpoint` / `FER_CHECKPOINT` at a `.pt` you bring. There is no hosted
  weight download.
- A hosted demo or “PRs welcome” support contract
- A claim that 7-class FER from a Haar crop is a production emotion detector
