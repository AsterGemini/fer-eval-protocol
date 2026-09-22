# fer-eval-protocol

**Contribution: an evaluation and data-merge protocol for 7-class facial
emotion recognition** — not another EfficientNet tutorial, and not a
plug-and-play emotion API.

Read this first: [`fer-system/PROTOCOL.md`](fer-system/PROTOCOL.md).


> **Personal practice:** built while applying lessons from Chip Huyen’s *Designing Machine Learning Systems*. Shared publicly in case the measurement protocol helps others — not a polished product or SOTA claim. The reusable piece is [`fer-system/PROTOCOL.md`](fer-system/PROTOCOL.md).

## What this is

Pinned measurement rules + a runnable public training recipe:

- **Frozen official FER val** (files and labels). No self-labeled val overlay.
- **Checkpoint / early-stop on** `raf_db.macro_f1` (eval-only RAF-DB val).
- **Near-dup filtering** when merging extra Kaggle packs into train.
- **Hashed public mix v1.4.0** (`data/manifest.json`) and default
  `configs/public.yaml`.
- **Code + docs** under [`fer-system/`](fer-system/).

The trainer is a standard EfficientNet (timm) fine-tune. Architecture is
commodity; the protocol is the reusable part.

## What this is not

- Not a SOTA leaderboard entry.
- Not a production emotion detector.
- **No face images and no trained weights** in git (dataset licenses).
  Gradio needs a local `.pt` you train or bring legally.

## Quick paths

| Goal | Start here |
|------|------------|
| Understand / reuse the protocol | [`fer-system/PROTOCOL.md`](fer-system/PROTOCOL.md) |
| Data roles + acknowledgments | [`fer-system/DATA_SOURCES.md`](fer-system/DATA_SOURCES.md), [`fer-system/README.md#acknowledgments`](fer-system/README.md#acknowledgments) |
| Smoke checks (no full train) | [`fer-system/VERIFY.md`](fer-system/VERIFY.md) |
| Train yourself | `cd fer-system && python -m data.download --config configs/public.yaml && python -m src.train --model b0` |
| Local Gradio demo | `python app/app.py --checkpoint /path/to/local.pt` |

## License

- Code: [MIT](LICENSE)
- Datasets: third-party (Kaggle / original authors). Not covered by the MIT
  license. Not in this repo.
