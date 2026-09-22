"""Prepare CK+48 lab faces as a 7-class ImageFolder (minority classes only).

The ``shawon10/ckplus`` pack is posed peak-expression frames with expert
labels. This hook copies **disgust** and **fear** only, drops ``contempt``,
and creates empty folders for the other five classes so ``FERDataset`` can
scan a 7-class tree. No val split is taken.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.utils import setup_logging

logger = logging.getLogger("fer.prepare_ckplus")

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
FOLDER_TO_CLASS = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sadness": "sad",
    "surprise": "surprise",
}
MIN_EXPECTED = {"disgust": 100, "fear": 50}


def find_ckplus48(raw_root: Path) -> Path:
    """Prefer the top-level ``CK+48`` tree (the pack also nests a duplicate)."""
    raw_root = Path(raw_root)
    direct = raw_root / "CK+48"
    if (direct / "disgust").is_dir() and (direct / "fear").is_dir():
        return direct
    hits = [
        p
        for p in raw_root.rglob("CK+48")
        if p.is_dir() and (p / "disgust").is_dir() and (p / "fear").is_dir()
    ]
    if not hits:
        raise FileNotFoundError(f"No CK+48/disgust+fear tree under {raw_root}")
    hits.sort(key=lambda p: len(p.parts))
    return hits[0]


def _iter_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_minority(src_root: Path, train_root: Path) -> dict[str, int]:
    """Copy disgust/fear into a 7-class ImageFolder; return counts."""
    train_root = Path(train_root)
    for name in ALL_CLASSES:
        (train_root / name).mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in ALL_CLASSES}
    for folder, canonical in FOLDER_TO_CLASS.items():
        if canonical not in TARGET_CLASSES:
            continue
        class_dir = src_root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing CK+ folder {class_dir}")
        for src in _iter_images(class_dir):
            dest = train_root / canonical / src.name
            shutil.copy2(src, dest)
            counts[canonical] += 1
    return counts


def prepare_ckplus(raw_root: Path, out_root: Path) -> Path:
    """Build ``<out_root>/train/<class>/`` from the CK+48 pack."""
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    src_root = find_ckplus48(raw_root)
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
                logger.info("CK+ already prepared at %s (counts=%s)", out_root, disk)
                return out_root
        logger.warning("CK+ prep stale at %s — rebuilding", out_root)
        shutil.rmtree(train_root)

    counts = copy_minority(src_root, train_root)
    meta = {
        "source": "shawon10/ckplus",
        "source_dir": str(src_root),
        "targets": list(TARGET_CLASSES),
        "counts": counts,
        "num_crops": sum(counts.values()),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("CK+ minority prepared: %d crops → %s | counts=%s", sum(counts.values()), train_root, counts)
    missing = [c for c in TARGET_CLASSES if counts[c] < MIN_EXPECTED[c]]
    if missing:
        raise RuntimeError(f"CK+ minority too small for {missing}: {counts}")
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CK+48 disgust/fear ImageFolder")
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    args = parser.parse_args()
    setup_logging("fer.prepare_ckplus")
    prepare_ckplus(Path(args.raw_root), Path(args.out_root))


if __name__ == "__main__":
    main()
