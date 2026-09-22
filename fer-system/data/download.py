"""Download and verify FER datasets via kagglehub (primary + optional extras)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils import load_config, project_root, setup_logging

logger = logging.getLogger("fer.download")

# Expected emotion class folder names in the cleaned FER2013-style dataset.
EXPECTED_CLASSES = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Bump when the training mix or val protocol changes:
#   MAJOR — class schema or primary val split changes
#   MINOR — new extra source added (e.g. AffectNet train, ExpW train)
#   PATCH — same sources, fingerprint refresh only
# Public default (configs/public.yaml) is mix v1.4.0.
# 1.3.0–1.5.0 minority-source experiments remain preparable.
# Committed data/manifest.json is the v1.4.0 public-mix SHA snapshot
# (FER2013 + RAF-DB + AffectNet + public ExpW + FER+/CK+/minority_boost).
MANIFEST_VERSION = "1.4.0"
MANIFEST_FILENAME = "manifest.json"


def download_dataset(kaggle_slug: str) -> Path:
    """Download (or resolve cached) dataset and return its root path.

    kagglehub caches downloads under ~/.cache/kagglehub by default, so
    repeated calls are cheap. We treat the returned path as the source of
    truth rather than copying tens of thousands of images into the repo.
    """
    import kagglehub

    logger.info("Downloading / resolving dataset: %s", kaggle_slug)
    path = Path(kagglehub.dataset_download(kaggle_slug))
    logger.info("Dataset root: %s", path)
    return path


def resolve_dataset_root(data_cfg: dict[str, Any]) -> Path:
    """Resolve a dataset root from ``root_dir`` or ``kaggle_slug``."""
    if data_cfg.get("root_dir"):
        root = Path(data_cfg["root_dir"])
        if not root.is_absolute():
            root = project_root() / root
        return root
    slug = data_cfg.get("kaggle_slug")
    if not slug:
        raise ValueError("Dataset source needs either root_dir or kaggle_slug")
    return download_dataset(str(slug))


def resolve_split_dir(dataset_root: Path, relative_split: str) -> Path:
    """Resolve a train/val split directory under the dataset root.

    The jonathanoheix dataset typically lays out as:
        <root>/images/train/<class>/*.jpg
        <root>/images/validation/<class>/*.jpg
    Some cache layouts nest an extra folder; we search a few candidates.
    """
    candidates = [
        dataset_root / relative_split,
        dataset_root / Path(relative_split).name,
    ]
    # Also try one level deeper if kagglehub wraps content.
    for child in dataset_root.iterdir() if dataset_root.is_dir() else []:
        if child.is_dir():
            candidates.append(child / relative_split)
            candidates.append(child / Path(relative_split).name)

    for cand in candidates:
        if cand.is_dir():
            return cand

    raise FileNotFoundError(
        f"Could not find split '{relative_split}' under {dataset_root}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def verify_split(
    split_dir: Path,
    class_names: list[str] | tuple[str, ...] = EXPECTED_CLASSES,
    folder_map: dict[str, str] | None = None,
) -> dict[str, int]:
    """Verify class folders exist and return per-class image counts.

    ``folder_map`` remaps canonical class names → on-disk folder names when they differ.
    """
    counts: dict[str, int] = {}
    missing = []
    folder_map = folder_map or {}
    for name in class_names:
        folder = folder_map.get(name, name)
        class_dir = split_dir / folder
        if not class_dir.is_dir():
            missing.append(f"{name} (folder={folder})")
            counts[name] = 0
            continue
        n = sum(1 for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        counts[name] = n

    if missing:
        raise FileNotFoundError(f"Missing class folders in {split_dir}: {missing}")

    total = sum(counts.values())
    logger.info("Split %s — %d images across %d classes", split_dir, total, len(counts))
    for name, n in counts.items():
        logger.info("  %-10s %6d", name, n)
    return counts


def get_dataset_paths(cfg: dict) -> tuple[Path, Path]:
    """Return primary (train_dir, val_dir) from config, downloading if needed."""
    data_cfg = cfg["data"]
    root = resolve_dataset_root(data_cfg)
    train_dir = resolve_split_dir(root, data_cfg["train_dir"])
    val_dir = resolve_split_dir(root, data_cfg["val_dir"])
    return train_dir, val_dir


def get_extra_train_sources(cfg: dict) -> list[dict[str, Any]]:
    """Resolve optional extra train-only sources (merged into training, not val).

    Each entry in ``data.extra_train_sources`` may specify:
      - name: short label for logs / metrics
      - kaggle_slug or root_dir
      - train_dir: relative split path
      - val_dir: optional relative split path for an eval-only second view
      - folder_map: optional {canonical_class: disk_folder}
      - prepare: optional prep hook for non-ImageFolder packs.
        ``"expw"`` extracts + crops faces. ``"ferplus_minority"`` joins
        official FER2013 pixels with FER+ crowd votes and keeps
        Training-only high-agreement disgust/fear. ``"ckplus"`` copies
        lab-posed CK+48 disgust/fear. ``"kdef"`` copies KDEF lab-posed
        disgust/fear. ``"minority_boost"`` keeps fer2013+affectnet
        disgusted/fearful crops that are not perceptual-hash
        near-duplicates of primary FER train+val, the existing AffectNet
        source, or RAF-DB train+val. ``"mma_minority"`` does the same
        filter on MMAFEDB **train** disgust/fear (valid/test unused —
        they contain FER2013 test splits). Prepared trees land in
        ``~/.cache/fer-system/<name>/`` (writable VM disk).
    """
    data_cfg = cfg["data"]
    extras_cfg = data_cfg.get("extra_train_sources") or []
    resolved: list[dict[str, Any]] = []
    for entry in extras_cfg:
        name = str(entry.get("name") or entry.get("kaggle_slug") or "extra")
        root = resolve_dataset_root(entry)
        if entry.get("prepare") == "expw":
            from data.prepare_expw import default_prepared_root, prepare_expw

            prep_log = logging.getLogger("fer.prepare_expw")
            prep_log.handlers = logger.handlers
            prep_log.setLevel(logger.level)
            prep_log.propagate = False
            root = prepare_expw(root, default_prepared_root(root, name))
        elif entry.get("prepare") == "ferplus_minority":
            from data.prepare_expw import default_prepared_root
            from data.prepare_ferplus_minority import prepare_ferplus_minority

            prep_log = logging.getLogger("fer.prepare_ferplus")
            prep_log.handlers = logger.handlers
            prep_log.setLevel(logger.level)
            prep_log.propagate = False
            root = prepare_ferplus_minority(root, default_prepared_root(root, name))
        elif entry.get("prepare") == "ckplus":
            from data.prepare_ckplus import prepare_ckplus
            from data.prepare_expw import default_prepared_root

            prep_log = logging.getLogger("fer.prepare_ckplus")
            prep_log.handlers = logger.handlers
            prep_log.setLevel(logger.level)
            prep_log.propagate = False
            root = prepare_ckplus(root, default_prepared_root(root, name))
        elif entry.get("prepare") == "minority_boost":
            from data.prepare_expw import default_prepared_root
            from data.prepare_minority_boost import prepare_minority_boost

            prep_log = logging.getLogger("fer.prepare_minority_boost")
            prep_log.handlers = logger.handlers
            prep_log.setLevel(logger.level)
            prep_log.propagate = False
            root = prepare_minority_boost(root, default_prepared_root(root, name))
        elif entry.get("prepare") == "kdef":
            from data.prepare_expw import default_prepared_root
            from data.prepare_kdef import prepare_kdef

            prep_log = logging.getLogger("fer.prepare_kdef")
            prep_log.handlers = logger.handlers
            prep_log.setLevel(logger.level)
            prep_log.propagate = False
            root = prepare_kdef(root, default_prepared_root(root, name))
        elif entry.get("prepare") == "mma_minority":
            from data.prepare_expw import default_prepared_root
            from data.prepare_mma_minority import prepare_mma_minority

            prep_log = logging.getLogger("fer.prepare_mma_minority")
            prep_log.handlers = logger.handlers
            prep_log.setLevel(logger.level)
            prep_log.propagate = False
            root = prepare_mma_minority(root, default_prepared_root(root, name))
        train_rel = entry.get("train_dir", "train")
        train_dir = resolve_split_dir(root, str(train_rel))
        val_rel = entry.get("val_dir")
        val_dir = resolve_split_dir(root, str(val_rel)) if val_rel else None
        folder_map = entry.get("folder_map") or None
        resolved.append(
            {
                "name": name,
                "kaggle_slug": entry.get("kaggle_slug"),
                "root_dir": str(root),
                "train_dir": train_dir,
                "val_dir": val_dir,
                "folder_map": dict(folder_map) if folder_map else None,
                "notes": entry.get("notes"),
            }
        )
        logger.info("Extra train source '%s' → %s", name, train_dir)
        if val_dir is not None:
            logger.info("Extra val source '%s' → %s (eval-only)", name, val_dir)
    return resolved


def describe_training_sources(cfg: dict) -> list[dict[str, Any]]:
    """Human/metrics-friendly list of datasets used for training vs validation."""
    data_cfg = cfg["data"]
    sources = [
        {
            "role": "primary_train_val",
            "name": data_cfg.get("source_name", "primary"),
            "kaggle_slug": data_cfg.get("kaggle_slug"),
            "train_dir": data_cfg.get("train_dir"),
            "val_dir": data_cfg.get("val_dir"),
            "notes": data_cfg.get("source_notes"),
        }
    ]
    for entry in data_cfg.get("extra_train_sources") or []:
        sources.append(
            {
                "role": "extra_train_only",
                "name": entry.get("name"),
                "kaggle_slug": entry.get("kaggle_slug"),
                "train_dir": entry.get("train_dir"),
                "val_dir": entry.get("val_dir"),
                "notes": entry.get("notes"),
            }
        )
    return sources


def _iter_image_files(split_dir: Path) -> list[Path]:
    """Sorted image paths under a class-folder tree."""
    files: list[Path] = []
    if not split_dir.is_dir():
        return files
    for path in sorted(split_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)
    return files


def hash_split_dir(split_dir: Path) -> str:
    """Content fingerprint of a split: sorted path + size + mtime (fast).

    Avoids hashing tens of thousands of image bytes on every verify while still
    detecting add/remove/replace. Paths are relative to ``split_dir``.
    """
    hasher = hashlib.sha256()
    for path in _iter_image_files(split_dir):
        rel = path.relative_to(split_dir).as_posix()
        stat = path.stat()
        line = f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}\n"
        hasher.update(line.encode("utf-8"))
    return hasher.hexdigest()


def _combine_hashes(hashes: list[str]) -> str:
    """Stable SHA256 over an ordered list of hex digests."""
    hasher = hashlib.sha256()
    for h in hashes:
        hasher.update(h.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def compute_dataset_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a versioned manifest of all resolved train/val/extra sources."""
    data_cfg = cfg["data"]
    class_names = list(cfg.get("class_names", list(EXPECTED_CLASSES)))
    sources: list[dict[str, Any]] = []
    all_hashes: list[str] = []

    train_dir, val_dir = get_dataset_paths(cfg)
    train_counts = verify_split(train_dir, class_names)
    val_counts = verify_split(val_dir, class_names)
    train_hash = hash_split_dir(train_dir)
    val_hash = hash_split_dir(val_dir)
    primary_hash = _combine_hashes([train_hash, val_hash])
    all_hashes.append(primary_hash)

    sources.append(
        {
            "name": data_cfg.get("source_name", "primary"),
            "role": "primary_train_val",
            "kaggle_slug": data_cfg.get("kaggle_slug"),
            "root": str(train_dir.parent.parent if "images" in str(train_dir) else train_dir.parent),
            "train_dir": str(train_dir),
            "val_dir": str(val_dir),
            "sha256": primary_hash,
            "train_sha256": train_hash,
            "val_sha256": val_hash,
            "train_counts": train_counts,
            "val_counts": val_counts,
            "counts": train_counts,
            "notes": data_cfg.get("source_notes"),
        }
    )

    for extra in get_extra_train_sources(cfg):
        folder_map = extra.get("folder_map")
        counts = verify_split(extra["train_dir"], class_names, folder_map=folder_map)
        split_hash = hash_split_dir(extra["train_dir"])
        all_hashes.append(split_hash)
        source_entry: dict[str, Any] = {
            "name": extra["name"],
            "role": "extra_train_only",
            "kaggle_slug": extra.get("kaggle_slug"),
            "root": extra.get("root_dir"),
            "train_dir": str(extra["train_dir"]),
            "val_dir": None,
            "sha256": split_hash,
            "counts": counts,
            "notes": extra.get("notes"),
        }
        if extra.get("val_dir") is not None:
            val_counts = verify_split(extra["val_dir"], class_names, folder_map=folder_map)
            val_hash = hash_split_dir(extra["val_dir"])
            all_hashes.append(val_hash)
            source_entry["val_dir"] = str(extra["val_dir"])
            source_entry["val_sha256"] = val_hash
            source_entry["val_counts"] = val_counts
        sources.append(source_entry)

    return {
        "version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fingerprint": _combine_hashes(all_hashes),
        "class_names": class_names,
        "sources": sources,
    }


def manifest_path() -> Path:
    """Default path for the committed data manifest."""
    return project_root() / "data" / MANIFEST_FILENAME


def write_manifest(cfg: dict[str, Any], path: Path | None = None) -> Path:
    """Compute and write ``data/manifest.json``; return the path written."""
    out = path or manifest_path()
    manifest = compute_dataset_manifest(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    logger.info(
        "Wrote data manifest version=%s fingerprint=%s → %s",
        manifest["version"],
        manifest["fingerprint"][:12],
        out,
    )
    return out


def load_manifest(path: Path | None = None) -> dict[str, Any] | None:
    """Load ``data/manifest.json`` if present; else return None."""
    p = path or manifest_path()
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest must be a mapping, got {type(data)}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify FER dataset(s)")
    parser.add_argument("--config", type=str, default="configs/public.yaml")
    args = parser.parse_args()

    setup_logging("fer.download")
    cfg = load_config(args.config)

    logger.info("Training sources: %s", describe_training_sources(cfg))
    # verify_split + hashing happen inside write_manifest / compute_dataset_manifest
    write_manifest(cfg)
    logger.info("Dataset(s) ready.")


if __name__ == "__main__":
    main()
