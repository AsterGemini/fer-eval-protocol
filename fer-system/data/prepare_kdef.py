"""Prepare KDEF lab faces as a 7-class ImageFolder (minority classes only).

The ``chenrich/kdef-database`` pack is already an ImageFolder of the
Karolinska Directed Emotional Faces (Lundqvist et al., 1998): 70 actors,
7 expressions, 3 camera angles (~420 images per class; full-profile
angles are not in this pack). Labels are expert-posed and disjoint from
FER2013 / RAF-DB / AffectNet, so there is no val-leakage risk.

This hook copies **disgust** and **fear** only and creates empty folders
for the other five classes so ``FERDataset`` can scan a 7-class tree.
No val split is taken; RAF-DB val stays the only checkpoint monitor.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.utils import setup_logging

logger = logging.getLogger("fer.prepare_kdef")

META_FILENAME = "prep_meta.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
ALL_CLASSES = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)
TARGET_CLASSES = ("disgust", "fear")
# Pack already uses canonical names; keep the map for a stray synonym.
FOLDER_TO_CLASS = {
    "angry": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "afraid": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}
MIN_EXPECTED = {"disgust": 300, "fear": 300}


def find_kdef_root(raw_root: Path) -> Path:
    """Locate the ImageFolder tree that holds KDEF disgust + fear folders."""
    raw_root = Path(raw_root)
    if (raw_root / "disgust").is_dir() and (raw_root / "fear").is_dir():
        return raw_root
    hits = [
        p.parent
        for p in raw_root.rglob("disgust")
        if p.is_dir() and (p.parent / "fear").is_dir()
    ]
    if not hits:
        raise FileNotFoundError(
            f"No KDEF ImageFolder with disgust+fear under {raw_root}"
        )
    hits.sort(key=lambda p: len(p.parts))
    return hits[0]


def _iter_images(class_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_minority(src_root: Path, train_root: Path) -> dict[str, int]:
    """Copy disgust/fear into a 7-class ImageFolder; return counts."""
    train_root = Path(train_root)
    for name in ALL_CLASSES:
        (train_root / name).mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in ALL_CLASSES}
    copied_canonical: set[str] = set()
    for folder, canonical in FOLDER_TO_CLASS.items():
        if canonical not in TARGET_CLASSES or canonical in copied_canonical:
            continue
        class_dir = src_root / folder
        if not class_dir.is_dir():
            continue
        for src in _iter_images(class_dir):
            dest = train_root / canonical / src.name
            shutil.copy2(src, dest)
            counts[canonical] += 1
        copied_canonical.add(canonical)
    missing = [c for c in TARGET_CLASSES if c not in copied_canonical]
    if missing:
        raise FileNotFoundError(
            f"Missing KDEF folders for {missing} under {src_root}"
        )
    return counts


def prepare_kdef(raw_root: Path, out_root: Path) -> Path:
    """Build ``<out_root>/train/<class>/`` from the KDEF ImageFolder pack."""
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    src_root = find_kdef_root(raw_root)
    train_root = out_root / "train"
    meta_path = out_root / META_FILENAME

    if meta_path.is_file() and train_root.is_dir():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        disk = {
            name: len(_iter_images(train_root / name)) if (train_root / name).is_dir() else 0
            for name in ALL_CLASSES
        }
        if meta.get("source_dir") == str(src_root) and meta.get("counts") == disk:
            if all(disk[c] >= MIN_EXPECTED[c] for c in TARGET_CLASSES):
                logger.info("KDEF already prepared at %s (counts=%s)", out_root, disk)
                return out_root
        logger.warning("KDEF prep stale at %s — rebuilding", out_root)
        shutil.rmtree(train_root)

    counts = copy_minority(src_root, train_root)
    meta = {
        "source": "chenrich/kdef-database",
        "source_dir": str(src_root),
        "targets": list(TARGET_CLASSES),
        "counts": counts,
        "num_crops": sum(counts.values()),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "KDEF minority prepared: %d crops → %s | counts=%s",
        sum(counts.values()),
        train_root,
        counts,
    )
    missing = [c for c in TARGET_CLASSES if counts[c] < MIN_EXPECTED[c]]
    if missing:
        raise RuntimeError(f"KDEF minority too small for {missing}: {counts}")
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KDEF disgust/fear ImageFolder")
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    args = parser.parse_args()
    setup_logging("fer.prepare_kdef")
    prepare_kdef(Path(args.raw_root), Path(args.out_root))


if __name__ == "__main__":
    main()
