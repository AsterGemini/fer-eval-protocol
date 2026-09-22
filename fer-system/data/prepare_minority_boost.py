"""Prepare near-dup-filtered disgust/fear crops from the fer2013+affectnet pack.

The ``prasadsomvanshih/fer2013-affectnet-dataset-emotions`` Kaggle pack merges
FER2013 with AffectNet-family images (48x48 grayscale). Its name is a leakage
hazard: measured with 16x16 average-hash (256-bit) on 48x48 grayscale,
its ``train/fearful`` contains ~895 near-duplicates of primary FER **val**
images and its ``train/disgusted`` ~97 (hamming <= 10; recompressed FER2013
images sit at distance 0-1). It also overlaps the existing AffectNet extra
source (~1.6k fear / ~1.3k disgust near-dups).

This hook keeps only ``disgusted``/``fearful`` crops that are NOT near-dups of:

- primary FER2013 train **and val** (val exclusion is a hard guarantee —
  that split is the frozen, literature-comparable monitor)
- the existing ``mstjebashazida/affectnet`` train source (avoid double weight)
- RAF-DB train + val (checkpoint monitor; measured 0 hits, kept as guard)

Kept crops land in ``<out_root>/train/{disgust,fear}/`` with the other five
class folders empty so ``FERDataset`` can scan a 7-class tree. Train only —
nothing here ever enters a val split.

Known limitation: ExpW is not in the exclusion set (it is only extracted on
the Colab VM). ExpW is a disjoint web collection (Zhang et al. 2016) and the
candidate pack showed 0 near-dups against RAF and 0 exact matches in prior
checks, so residual ExpW overlap risk is low.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from src.utils import setup_logging

logger = logging.getLogger("fer.prepare_minority_boost")

META_FILENAME = "prep_meta.json"
HASH_CACHE_FILENAME = "exclusion_hashes.npz"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALL_CLASSES = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)
# On-disk folder -> canonical class. Only these two folders are read.
FOLDER_TO_CLASS = {"disgusted": "disgust", "fearful": "fear"}
# 256-bit aHash; recompressed duplicates of the same face land at distance
# 0-1, different faces cluster near ~128. 10 leaves margin for recompression
# and crop jitter while being far from any same-person different-photo pair.
DEFAULT_MAX_HAMMING = 10
AHASH_SIZE = 16
HASH_BYTES = (AHASH_SIZE * AHASH_SIZE) // 8
MATCH_CHUNK = 128
# Post-filter floors; measured yields are ~1.15k disgust / ~1.46k fear after
# excluding FER2013 (train+val) and existing-AffectNet near-duplicates.
MIN_EXPECTED = {"disgust": 1000, "fear": 1400}

# Exclusion sources resolved via kagglehub cache. ``fer2013_primary`` protects
# the frozen val split; failure to resolve it must abort, never silently skip.
EXCLUSION_SOURCES = (
    {
        "key": "fer2013_primary",
        "kaggle_slug": "jonathanoheix/face-expression-recognition-dataset",
        "splits": ("train", "validation"),
        "mandatory": True,
    },
    {
        "key": "affectnet_mstjebashazida",
        "kaggle_slug": "mstjebashazida/affectnet",
        "splits": ("Train",),
        "mandatory": True,
    },
    {
        "key": "raf_db",
        "kaggle_slug": "nishchalchandel/raf-db-face-emotion-dataset",
        "splits": ("train", "validation"),
        "mandatory": True,
    },
)

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def ahash_bytes(path: Path, size: int = AHASH_SIZE) -> np.ndarray | None:
    """Average-hash of an image as ``(HASH_BYTES,)`` uint8 (size*size bits)."""
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size, size), Image.BILINEAR)
            arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
        bits = (arr > arr.mean()).astype(np.uint8)
        return np.packbits(bits)
    except Exception as exc:
        logger.debug("ahash failed for %s: %s", path, exc)
        return None


def iter_image_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def hash_tree_bytes(root: Path) -> tuple[np.ndarray, list[str]]:
    """aHash every image under ``root``; return ``(N,32) uint8`` + paths."""
    hashes: list[np.ndarray] = []
    paths: list[str] = []
    for p in iter_image_files(root):
        h = ahash_bytes(p)
        if h is not None:
            hashes.append(h)
            paths.append(str(p))
    if not hashes:
        return np.zeros((0, HASH_BYTES), dtype=np.uint8), []
    return np.stack(hashes), paths


def hamming_min(candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Min hamming distance of each candidate row to any reference row."""
    if reference.shape[0] == 0:
        return np.full(candidates.shape[0], 1 << 15, dtype=np.int32)
    out = np.empty(candidates.shape[0], dtype=np.int32)
    for start in range(0, candidates.shape[0], MATCH_CHUNK):
        chunk = candidates[start : start + MATCH_CHUNK]
        xor = np.bitwise_xor(chunk[:, None, :], reference[None, :, :])
        dist = _POPCOUNT_LUT[xor].sum(axis=-1)
        out[start : start + chunk.shape[0]] = dist.min(axis=1)
    return out


def tree_fingerprint(root: Path) -> str:
    """Cheap change detector for an exclusion tree (count + bytes + root)."""
    files = iter_image_files(root)
    total = sum(p.stat().st_size for p in files)
    return f"{root}|{len(files)}|{total}"


def find_pack_root(raw_root: Path) -> Path:
    """Locate the ``fer2013+affectnet`` tree (tolerates kagglehub nesting)."""
    raw_root = Path(raw_root)
    direct = raw_root / "fer2013+affectnet"
    if (direct / "train" / "disgusted").is_dir():
        return direct
    hits = [
        p
        for p in raw_root.rglob("train")
        if p.is_dir() and (p / "disgusted").is_dir() and (p / "fearful").is_dir()
    ]
    if not hits:
        raise FileNotFoundError(
            f"No train/disgusted+fearful tree under {raw_root} "
            "(expected the fer2013+affectnet ImageFolder pack)"
        )
    hits.sort(key=lambda p: len(p.parts))
    return hits[0].parent


def _resolve_split_dir(root: Path, split: str) -> Path | None:
    """Find ``<root>/<split>`` tolerating one level of kagglehub nesting."""
    from data.download import resolve_split_dir

    try:
        return resolve_split_dir(Path(root), split)
    except FileNotFoundError:
        return None


def default_exclusion_dirs(
    sources: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve exclusion trees via kagglehub; fail closed on mandatory sources."""
    dirs: dict[str, dict[str, Any]] = {}
    for spec in sources if sources is not None else EXCLUSION_SOURCES:
        key = str(spec["key"])
        try:
            import kagglehub

            root = Path(kagglehub.dataset_download(str(spec["kaggle_slug"])))
            split_dirs = []
            for split in spec["splits"]:
                found = _resolve_split_dir(root, str(split))
                if found is None:
                    raise FileNotFoundError(f"{key}: split {split!r} not under {root}")
                split_dirs.append(found)
        except Exception as exc:
            if spec.get("mandatory"):
                raise RuntimeError(
                    f"Cannot resolve mandatory exclusion source {key!r}: {exc}. "
                    "Refusing to prepare without leakage guards."
                ) from exc
            logger.warning("Skipping optional exclusion source %s: %s", key, exc)
            continue
        fingerprint = _combine_fingerprints(split_dirs)
        dirs[key] = {"dirs": split_dirs, "fingerprint": fingerprint}
        logger.info("Exclusion source %s: %s", key, [str(d) for d in split_dirs])
    return dirs


def _combine_fingerprints(dirs: list[Path]) -> str:
    parts = [tree_fingerprint(d) for d in dirs]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def load_or_build_exclusion_hashes(
    exclusion: dict[str, dict[str, Any]],
    cache_path: Path,
) -> dict[str, np.ndarray]:
    """Per-source aHash arrays, cached on disk keyed by tree fingerprint."""
    cached: dict[str, Any] = {}
    if Path(cache_path).is_file():
        blob = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(blob["meta_json"].tobytes().decode("utf-8")))
        cached = {k: blob[f"hash_{k}"] for k in meta.get("sources", {})}
        fps = meta.get("sources", {})
        if all(
            k in cached and fps.get(k) == spec["fingerprint"]
            for k, spec in exclusion.items()
        ):
            logger.info("Reusing cached exclusion hashes: %s", cache_path)
            return cached
        logger.info("Exclusion hash cache stale — rebuilding")

    out: dict[str, np.ndarray] = {}
    for key, spec in exclusion.items():
        arrays = [hash_tree_bytes(d)[0] for d in spec["dirs"]]
        out[key] = (
            np.concatenate(arrays) if arrays else np.zeros((0, HASH_BYTES), dtype=np.uint8)
        )
        logger.info("Exclusion hashes %s: %d images", key, out[key].shape[0])

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "sources": {k: spec["fingerprint"] for k, spec in exclusion.items()},
        "ahash_size": AHASH_SIZE,
    }
    np.savez(
        cache_path,
        meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        **{f"hash_{k}": v for k, v in out.items()},
    )
    return out


def select_crops_from_class_dirs(
    class_dirs: dict[str, Path],
    exclusion_hashes: dict[str, np.ndarray],
    *,
    max_hamming: int = DEFAULT_MAX_HAMMING,
    log_prefix: str = "filtered",
) -> tuple[dict[str, list[Path]], dict[str, dict[str, int]]]:
    """Keep images whose aHash is far from every exclusion source.

    ``class_dirs`` maps canonical class name → on-disk folder. One
    representative is kept per unique aHash inside each folder.
    """
    kept: dict[str, list[Path]] = {name: [] for name in class_dirs}
    accounting: dict[str, dict[str, int]] = {}
    for canonical, class_dir in class_dirs.items():
        class_dir = Path(class_dir)
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing pack folder {class_dir}")
        files = iter_image_files(class_dir)
        hashes, paths = hash_tree_bytes(class_dir)
        # One representative per unique aHash inside the pack itself.
        _, unique_idx = np.unique(hashes, axis=0, return_index=True)
        order = sorted(unique_idx)
        hashes = hashes[order]
        paths = [paths[i] for i in order]
        acct: dict[str, int] = {
            "files_in": len(files),
            "unreadable": len(files) - len(paths),
            "unique_candidates": int(hashes.shape[0]),
        }
        alive = np.ones(hashes.shape[0], dtype=np.bool_)
        for key, ref in exclusion_hashes.items():
            if not alive.any():
                acct[f"excluded_{key}"] = 0
                continue
            dists = hamming_min(hashes[alive], ref)
            drop = dists <= max_hamming
            acct[f"excluded_{key}"] = int(drop.sum())
            alive[np.where(alive)[0][drop]] = False
        acct["kept"] = int(alive.sum())
        kept[canonical] = [Path(paths[i]) for i in np.where(alive)[0]]
        accounting[canonical] = acct
        logger.info(
            "%s %s: in=%d unique=%d kept=%d | %s",
            log_prefix,
            canonical,
            acct["files_in"],
            acct["unique_candidates"],
            acct["kept"],
            {k: v for k, v in acct.items() if k.startswith("excluded_")},
        )
    return kept, accounting


def select_new_crops(
    pack_root: Path,
    exclusion_hashes: dict[str, np.ndarray],
    *,
    max_hamming: int = DEFAULT_MAX_HAMMING,
) -> tuple[dict[str, list[Path]], dict[str, dict[str, int]]]:
    """Pick disgusted/fearful crops that near-dup-match no exclusion source.

    Returns ``(kept, accounting)`` where accounting breaks down, per canonical
    class, how many candidates each exclusion source rejected.
    """
    class_dirs: dict[str, Path] = {}
    for folder, canonical in FOLDER_TO_CLASS.items():
        class_dir = Path(pack_root) / "train" / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing pack folder {class_dir}")
        class_dirs[canonical] = class_dir
    return select_crops_from_class_dirs(
        class_dirs,
        exclusion_hashes,
        max_hamming=max_hamming,
        log_prefix="minority_boost",
    )


def write_imagefolder(kept: dict[str, list[Path]], train_root: Path) -> dict[str, int]:
    train_root = Path(train_root)
    for name in ALL_CLASSES:
        (train_root / name).mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in ALL_CLASSES}
    for canonical, paths in kept.items():
        for i, src in enumerate(paths):
            dest = train_root / canonical / f"{i:05d}_{src.name}"
            shutil.copy2(src, dest)
            counts[canonical] += 1
    return counts


def prepare_minority_boost(
    raw_root: Path,
    out_root: Path,
    *,
    max_hamming: int = DEFAULT_MAX_HAMMING,
    exclusion_resolver: Callable[[], dict[str, dict[str, Any]]] | None = None,
) -> Path:
    """Build ``<out_root>/train/<class>/`` from the fer2013+affectnet pack.

    ``exclusion_resolver`` is injectable for tests; production resolves the
    mandatory exclusion sources from the kagglehub cache.
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    pack_root = find_pack_root(raw_root)
    train_root = out_root / "train"
    meta_path = out_root / META_FILENAME

    params = {
        "max_hamming": int(max_hamming),
        "ahash_size": AHASH_SIZE,
        "pack_root": str(pack_root),
        "exclusion_sources": [str(s["key"]) for s in EXCLUSION_SOURCES],
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
            and all(disk[c] >= MIN_EXPECTED[c] for c in FOLDER_TO_CLASS.values())
        ):
            logger.info("minority_boost already prepared at %s (counts=%s)", out_root, disk)
            return out_root
        logger.warning("minority_boost prep stale at %s — rebuilding", out_root)
        shutil.rmtree(train_root)

    resolver = exclusion_resolver or default_exclusion_dirs
    exclusion = resolver()
    exclusion_hashes = load_or_build_exclusion_hashes(
        exclusion, out_root / HASH_CACHE_FILENAME
    )
    kept, accounting = select_new_crops(
        pack_root, exclusion_hashes, max_hamming=max_hamming
    )
    counts = write_imagefolder(kept, train_root)
    meta = {
        "source": "prasadsomvanshih/fer2013-affectnet-dataset-emotions",
        "pack_root": str(pack_root),
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
        "minority_boost prepared: %d crops → %s | counts=%s",
        sum(counts.values()),
        train_root,
        counts,
    )
    missing = [c for c in FOLDER_TO_CLASS.values() if counts[c] < MIN_EXPECTED[c]]
    if missing:
        raise RuntimeError(
            f"minority_boost produced too few crops for {missing}: {counts} "
            f"(accounting={accounting})"
        )
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare near-dup-filtered disgust/fear crops (train only)"
    )
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--max-hamming", type=int, default=DEFAULT_MAX_HAMMING)
    args = parser.parse_args()
    setup_logging("fer.prepare_minority_boost")
    prepare_minority_boost(
        Path(args.raw_root),
        Path(args.out_root),
        max_hamming=args.max_hamming,
    )


if __name__ == "__main__":
    main()
