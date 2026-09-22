# Training data sources

> **Acknowledgments:** original dataset authors are credited in [README.md § Acknowledgments](README.md#acknowledgments). Kaggle packs below are redistribution hosts — verify each license.


Documented mix for the public protocol ([PROTOCOL.md](PROTOCOL.md)).
`configs/public.yaml` is the default train recipe (public Kaggle slugs).
Private ExpW packs and run-specific recipes belong in a gitignored local
file such as `configs/private.yaml` — not in this snapshot.

The public recipe is mix **v1.4.0**. That version lives in
`data/download.py` (`MANIFEST_VERSION`) and in the committed
`data/manifest.json` SHA snapshot (FER2013 + RAF-DB + AffectNet + public ExpW
+ FER+/CK+/`minority_boost`). Later minority packs remain preparable; `kdef`
and `mma_minority` stay out of the default recipe.

| Manifest | Mix |
|----------|-----|
| 1.0.0 | FER2013 train/val + RAF-DB train (later runs also hashed RAF-DB val) |
| 1.1.0 | + AffectNet train (~14.5k, 7-class; `contempt` ignored) |
| 1.2.0 fingerprint | + ExpW train (in-the-wild face crops from `label.lst` boxes) |
| 1.3.0 experimental | + FER+ Training-only high-agreement disgust/fear + CK+48 |
| **1.4.0 public.yaml default** | val-label overlay **reverted** (frozen official FER val); + `minority_boost` |
| 1.5.0 experimental | + `kdef` + `mma_minority` (not in the default recipe) |

Primary FER2013 val **files and labels** are unchanged across versions and now
fully frozen: the 25-error human-audit overlay (`data.val_label_corrections`)
was reverted on 2026-08-24. It remapped 13 "wrong" official labels to the
audit-time *model* prediction — circular evaluation (the human verdicts never
named a replacement class, and the same model was confidently wrong on 8 of 21
decided audit cases). The audit artifacts stay under `data/label_review/` as
documentation. All FER val numbers are again official-label numbers, comparable
to the literature and to the ~71% label-noise ceiling discussion.
Checkpointing still uses RAF-DB val (`raf_db.macro_f1`).

| Role | Name | Kaggle slug | Split used | Classes |
|------|------|-------------|------------|---------|
| Primary train + val | `fer2013_jonathanoheix` | [`jonathanoheix/face-expression-recognition-dataset`](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) | `images/train`, `images/validation` | angry, disgust, fear, happy, neutral, sad, surprise |
| Extra train + eval-only val | `raf_db` | [`nishchalchandel/raf-db-face-emotion-dataset`](https://www.kaggle.com/datasets/nishchalchandel/raf-db-face-emotion-dataset) | `train`, `validation` | same 7 (RAF-DB basic emotions) |
| Extra train only | `affectnet` | [`mstjebashazida/affectnet`](https://www.kaggle.com/datasets/mstjebashazida/affectnet) | `Train` | same 7 (`anger` folder remapped; `contempt` ignored) |
| Extra train only | `expw` | [`shahzadabbas/expression-in-the-wild-expw-dataset`](https://www.kaggle.com/datasets/shahzadabbas/expression-in-the-wild-expw-dataset) | prepared `train` (see below) | same 7 (numeric labels 0–6 mapped) |
| Extra train only | `ferplus_minority` | [`deadskull7/fer2013`](https://www.kaggle.com/datasets/deadskull7/fer2013) + [FER+](https://github.com/microsoft/FERPlus) | prepared `train` (Training rows only) | disgust, fear (other class folders empty) |
| Extra train only | `ckplus` | [`shawon10/ckplus`](https://www.kaggle.com/datasets/shawon10/ckplus) | prepared `train` (CK+48 disgust/fear) | disgust, fear (contempt dropped; no val) |
| Extra train only | `minority_boost` | [`prasadsomvanshih/fer2013-affectnet-dataset-emotions`](https://www.kaggle.com/datasets/prasadsomvanshih/fer2013-affectnet-dataset-emotions) | prepared `train` (near-dup-filtered disgusted/fearful) | disgust, fear (other class folders empty; no val) |
| Extra train only | `kdef` | [`chenrich/kdef-database`](https://www.kaggle.com/datasets/chenrich/kdef-database) | prepared `train` (KDEF disgust/fear) | disgust, fear (other class folders empty; no val) |
| Extra train only | `mma_minority` | [`mahmoudima/mma-facial-expression`](https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression) | prepared `train` (near-dup-filtered MMAFEDB train disgust/fear) | disgust, fear (valid/test unused) |

## Why ExpW (train only)

ExpW (~92k manually labeled in-the-wild faces, Zhang et al. 2016) is a
different collection from FER2013 / RAF-DB / AffectNet, so it adds diversity
without remixing an existing pool. It is the largest single addition so far:
the train mix goes from 55,641 → 144,241 images.

Prepared per-class counts (88,600 unique face crops; 3,193 duplicate
image/face-id label rows collapse to the same file):

| angry | disgust | fear | happy | neutral | sad | surprise |
|-------|---------|------|-------|---------|-----|----------|
| 3,602 | 3,805 | 1,064 | 28,893 | 33,927 | 10,429 | 6,880 |

ExpW is happy/neutral-heavy, so the combined mix leans harder on
`use_class_weights` (auto inverse-frequency). Notably, disgust — the rare
primary class (436) — gains 3,805 examples, and sad roughly doubles.

The Kaggle pack used by `configs/public.yaml` is **not** ImageFolder: it ships split 7z volumes plus
`label.lst` (per-face boxes + numeric labels). `data/prepare_expw.py`
concatenates the volumes, extracts with a native 7z CLI (`7zz` / `7z` /
`7za`; installs `p7zip-full` via apt-get on Colab if missing), crops each
labeled face, and writes ``~/.cache/fer-system/expw/train/<class>/`` on the VM disk
(never ``/kaggle/input`` — that mount is read-only — and never Drive).
Override with `FER_PREPARED_ROOT`. Prep is
idempotent only when on-disk counts match `prep_meta.json` and all 7 classes
are non-empty. `bash scripts/train.sh` is enough: download + prep run inside
training. No val split is taken from ExpW: RAF-DB val remains the only
checkpoint monitor.

## Why RAF-DB (not another FER2013 mirror)

`ananthu017/emotion-detection-fer` and similar packs are FER2013 redistributions.
Merging them mostly duplicates the primary set and risks train/val leakage.

RAF-DB is a different real-world collection (~12k train images in this Kaggle
pack) with the same seven basic emotions, so it adds diversity without remixing
the same FER2013 pool.

## Why AffectNet (train only)

AffectNet is in-the-wild (same family as RAF-DB, our checkpoint monitor). The
Kaggle pack `mstjebashazida/affectnet` is an ImageFolder subset (~14.5k train
images across the 7 classes after dropping `contempt`). Folder name `anger` is
remapped to canonical `angry` via `folder_map`.

The pack's `Test` split is **not** used: RAF-DB val remains the selection
signal, and `Test` folder casing is inconsistent (`Anger` vs `anger`). Leave it
as an unused holdout.

## Why FER+ minority (train only)

FER2013's original single labels are noisy, especially on disgust and fear.
[FER+](https://github.com/microsoft/FERPlus) (Barsoum et al., 2016) re-labels
each official FER2013 row with 10 crowd votes. `data/prepare_ferplus_minority.py`
keeps a row only when:

- `Usage` is **Training** (PublicTest / PrivateTest never enter the mix)
- the majority among the 7 mapped emotions is `disgust` or `fear`
- that majority has **≥ 6 / 10** votes and is not a tie
- `unknown + NF` is at most 2

The official CSV (`deadskull7/fer2013`) supplies pixels; FER+ votes come from
Microsoft's `fer2013new.csv`. Crops land in
``~/.cache/fer-system/ferplus_minority/train/``. Other class folders exist but
stay empty so `FERDataset` can scan a 7-class tree.

Prepared counts at ≥6/10 votes (Training only): **disgust 45**, **fear 338**.
These faces can overlap the primary FER train split. That is intentional:
the extra copy carries the cleaner crowd label. They must not overlap
primary val, which is why PublicTest/PrivateTest are dropped.

## Why CK+48 minority (train only)

CK+48 is lab-posed peak frames with expert labels — small, but much cleaner
than in-the-wild scrapes. `data/prepare_ckplus.py` copies **disgust (177)**
and **fear (75)** from the top-level `CK+48/` tree (the pack also nests a
duplicate under `ck/CK+48/`; that copy is ignored). `contempt` is dropped.
There is no CK+ val view: RAF-DB val stays the only checkpoint monitor.

## Why minority_boost (train only)

FER2013 disgust is tiny (436 train) and both weak RAF classes are
disgust/fear. The `prasadsomvanshih/fer2013-affectnet-dataset-emotions` pack
has 2,842 disgusted / 7,148 fearful train crops — but its own name advertises
the hazard: it merges FER2013, and perceptual-hash measurement (16×16 aHash,
hamming ≤ 10 on 48×48 grayscale) confirms **~895 fearful and ~97 disgusted
crops are near-duplicates of primary FER val images**, plus ~3.2k/~0.3k of
FER train and ~1.6k/~1.3k of the existing AffectNet source. Adding it raw
would leak the frozen val split into training and double-weight existing
sources.

`data/prepare_minority_boost.py` therefore keeps a crop only when its aHash
is > 10 bits from every image in: primary FER train **and** val (hard
guarantee — the loader aborts if this source cannot be resolved), the
existing `mstjebashazida/affectnet` train, and RAF-DB train + val (0 hits
measured; kept as a guard). Recompressed duplicates sit at distance 0–1;
different faces cluster near ~128, so the ≤10 threshold is conservative.
Exclusion hashes are cached in `~/.cache/fer-system/minority_boost/
exclusion_hashes.npz`, keyed by source-tree fingerprints. ExpW is not in the
exclusion set (it is extracted only on the Colab VM; it is a disjoint web
collection — measured 0 exact hits in prior checks, residual risk low).

Kept yields and per-source exclusion counts are recorded in
`~/.cache/fer-system/minority_boost/prep_meta.json`. Train only; no val view.

## Why KDEF (train only)

RAF-DB val F1 is weakest on disgust (~0.66) and fear (~0.76). CK+48 adds
only 177/75 lab frames. KDEF (Lundqvist et al., 1998) is a different
posed collection: 70 actors × 7 expressions × 3 angles in the
`chenrich/kdef-database` ImageFolder (~420 per class; full-profile
angles are not in this pack). It does not remix FER2013, so it is safe
to copy without a perceptual-hash filter.

`data/prepare_kdef.py` copies **disgust** and **fear** only. Train only;
RAF-DB val stays the checkpoint monitor.

JAFFE was considered and skipped: its terms of use forbid redistribution
via Kaggle / Colab. Mixed packs such as `stevemarcelloliem/kdef-raf-fer`
were skipped because their FER test split matches primary val counts
(disgust 111) and would leak if used as train.

## Why MMA minority (train only)

MMAFEDB (`mahmoudima/mma-facial-expression`) is the largest remaining
public 7-class ImageFolder with extra disgust (~3,231 train) and fear
(~4,859 train). Independent counts show the train split mixes **28,709
RGB crops — exactly FER2013 train size** — with extra grayscale faces,
and valid/test each have **3,589 RGB** crops (FER2013 PublicTest /
PrivateTest). Adding valid or test would leak the frozen primary FER val
split.

`data/prepare_mma_minority.py` therefore reads **train only**, keeps
disgust/fear, and applies the same 16×16 aHash ≤ 10 filter as
`minority_boost` against: primary FER train **and** val, existing
AffectNet, RAF-DB train+val, and the `fer2013+affectnet` pack already
in the mix (avoid double-weighting). Fear extras after subtracting FER
train copies are thin (~756), so a zero unique-fear yield does not abort
prep; disgust must still keep ≥ 200 unique crops. Yields land in
`~/.cache/fer-system/mma_minority/prep_meta.json`.

## How merging works

1. Download primary + each `data.extra_train_sources` entry via kagglehub.
   Entries with `prepare` (`expw`, `ferplus_minority`, `ckplus`,
   `kdef`, `minority_boost`, `mma_minority`) are materialized into
   ``~/.cache/fer-system/<name>/`` first (writable VM disk).
2. Build one `FERDataset` per train root (shared train transforms).
3. `ConcatDataset` → single training loader; weighted sampler uses combined labels.
4. Validation loader uses **only** the primary `val_dir` with official labels
   (`data.val_label_corrections` is `null` since 1.4.0 and must stay `null`;
   see the reverted-overlay note above).

Per-run `artifacts/models/efficientnet_b0/figures/metrics.json` records `training_sources` and
`training_source_counts` under `metrics`.
