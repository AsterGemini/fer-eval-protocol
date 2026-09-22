"""Prepare near-dup-filtered MMAFEDB train disgust/fear crops.

The ``mahmoudima/mma-facial-expression`` pack (MMAFEDB) is a merged
in-the-wild collection: ~93k train images across 7 classes. Independent
counts show the train split mixes **28,709 RGB crops (exactly FER2013
train size)** with extra grayscale faces, and the valid/test RGB counts
are **3,589** — FER2013 PublicTest / PrivateTest. Using valid or test
would leak the frozen primary FER val split.

This hook therefore:

- reads **train only** (valid/test are never opened as candidates)
- keeps only ``disgust`` and ``fear``
- drops a crop whose 16×16 aHash is ≤ 10 bits from primary FER train
  **or val**, the existing AffectNet extra source, RAF-DB train+val, or
  the ``fer2013+affectnet`` pack already used by ``minority_boost``

Kept crops land in ``<out_root>/train/{disgust,fear}/`` with the other
five class folders empty. Train only — nothing here enters a val split.

Fear extras in MMA train are thin once FER2013 copies are removed
(~4,859 − 4,103), so the fear floor is 0: unique fear is kept when it
survives the hash filter, but a zero yield does not abort the disgust
boost.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from data.prepare_minority_boost import (
    AHASH_SIZE,
    ALL_CLASSES,
    DEFAULT_MAX_HAMMING,
    EXCLUSION_SOURCES,
    default_exclusion_dirs,
    find_pack_root,
    iter_image_files,
    load_or_build_exclusion_hashes,
    select_crops_from_class_dirs,
    tree_fingerprint,
    write_imagefolder,
)
from src.utils import setup_logging

logger = logging.getLogger("fer.prepare_mma_minority")

META_FILENAME = "prep_meta.json"
HASH_CACHE_FILENAME = "exclusion_hashes.npz"
TARGET_CLASSES = ("disgust", "fear")
# MMA train extras after subtracting FER2013 train copies are large for
# disgust (~2.8k) and small for fear (~0.8k); both then lose AffectNet
# / minority_boost near-dups. Disgust must still yield a real boost.
MIN_EXPECTED = {"disgust": 200, "fear": 0}
# valid/test contain FER2013 PublicTest/PrivateTest as RGB 3589-image splits.
_FORBIDDEN_SPLIT_NAMES = frozenset({"valid", "validation", "val", "test"})
MERGED_PACK_SLUG = "prasadsomvanshih/fer2013-affectnet-dataset-emotions"


def find_mma_train(raw_root: Path) -> Path:
    """Locate ``MMAFEDB/train``; never return valid/test trees."""
    raw_root = Path(raw_root)
    direct = raw_root / "MMAFEDB" / "train"
    if (direct / "disgust").is_dir() and (direct / "fear").is_dir():
        return direct

    def _is_forbidden(path: Path) -> bool:
        return any(part.lower() in _FORBIDDEN_SPLIT_NAMES for part in path.parts)

    hits = [
        p
        for p in raw_root.rglob("train")
        if p.is_dir()
        and p.name.lower() == "train"
        and not _is_forbidden(p.parent)
        and (p / "disgust").is_dir()
        and (p / "fear").is_dir()
    ]
    if not hits:
        raise FileNotFoundError(
            f"No MMAFEDB/train/disgust+fear tree under {raw_root} "
            "(valid/test are ignored — they contain FER2013 test splits)"
        )
    hits.sort(key=lambda p: (0 if "MMAFEDB" in p.parts else 1, len(p.parts)))
    return hits[0]


def default_mma_exclusion_dirs() -> dict[str, dict[str, Any]]:
    """FER / AffectNet / RAF plus the existing fer2013+affectnet extra pack."""
    dirs = default_exclusion_dirs()
    try:
        import kagglehub

        root = Path(kagglehub.dataset_download(MERGED_PACK_SLUG))
        pack = find_pack_root(root)
        train = pack / "train"
        if not train.is_dir():
            raise FileNotFoundError(f"No train split under {pack}")
        dirs["fer2013_affectnet_pack"] = {
            "dirs": [train],
            "fingerprint": tree_fingerprint(train),
        }
        logger.info("Exclusion source fer2013_affectnet_pack: %s", train)
    except Exception as exc:
        raise RuntimeError(
            "Cannot resolve mandatory exclusion source "
            f"'fer2013_affectnet_pack': {exc}. Refusing to prepare "
            "mma_minority without the minority_boost pack guard."
        ) from exc
    return dirs


def prepare_mma_minority(
    raw_root: Path,
    out_root: Path,
    *,
    max_hamming: int = DEFAULT_MAX_HAMMING,
    exclusion_resolver: Callable[[], dict[str, dict[str, Any]]] | None = None,
) -> Path:
    """Build ``<out_root>/train/<class>/`` from MMAFEDB train disgust/fear."""
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    train_src = find_mma_train(raw_root)
    train_root = out_root / "train"
    meta_path = out_root / META_FILENAME

    params = {
        "max_hamming": int(max_hamming),
        "ahash_size": AHASH_SIZE,
        "train_src": str(train_src),
        "exclusion_sources": [str(s["key"]) for s in EXCLUSION_SOURCES]
        + ["fer2013_affectnet_pack"],
    }
    if meta_path.is_file() and train_root.is_dir():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        disk = {
            name: len(iter_image_files(train_root / name)) if (train_root / name).is_dir() else 0
            for name in ALL_CLASSES
        }
        if (
            meta.get("params") == params
            and meta.get("counts") == disk
            and all(disk[c] >= MIN_EXPECTED[c] for c in TARGET_CLASSES)
        ):
            logger.info("mma_minority already prepared at %s (counts=%s)", out_root, disk)
            return out_root
        logger.warning("mma_minority prep stale at %s — rebuilding", out_root)
        shutil.rmtree(train_root)

    resolver = exclusion_resolver or default_mma_exclusion_dirs
    exclusion = resolver()
    exclusion_hashes = load_or_build_exclusion_hashes(
        exclusion, out_root / HASH_CACHE_FILENAME
    )
    class_dirs = {name: train_src / name for name in TARGET_CLASSES}
    kept, accounting = select_crops_from_class_dirs(
        class_dirs,
        exclusion_hashes,
        max_hamming=max_hamming,
        log_prefix="mma_minority",
    )
    counts = write_imagefolder(kept, train_root)
    meta = {
        "source": "mahmoudima/mma-facial-expression",
        "train_src": str(train_src),
        "params": params,
        "counts": counts,
        "num_crops": sum(counts.values()),
        "accounting": accounting,
        "exclusion_fingerprints": {
            k: spec["fingerprint"] for k, spec in exclusion.items()
        },
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "mma_minority prepared: %d crops → %s | counts=%s",
        sum(counts.values()),
        train_root,
        counts,
    )
    missing = [c for c in TARGET_CLASSES if counts[c] < MIN_EXPECTED[c]]
    if missing:
        raise RuntimeError(
            f"mma_minority produced too few crops for {missing}: {counts} "
            f"(accounting={accounting})"
        )
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare near-dup-filtered MMAFEDB train disgust/fear"
    )
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--max-hamming", type=int, default=DEFAULT_MAX_HAMMING)
    args = parser.parse_args()
    setup_logging("fer.prepare_mma_minority")
    prepare_mma_minority(
        Path(args.raw_root),
        Path(args.out_root),
        max_hamming=args.max_hamming,
    )


if __name__ == "__main__":
    main()
