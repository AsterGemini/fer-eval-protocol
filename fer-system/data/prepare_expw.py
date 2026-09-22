"""Prepare the raw ExpW Kaggle pack into an ImageFolder tree.

The pack (``shahzadabbas/expression-in-the-wild-expw-dataset``) ships as
split 7z volumes (``origin.7z.*``) plus ``label.lst`` with per-face boxes:

    image_name face_id top left right bottom confidence expression_label

7z split volumes are plain byte splits, so concatenating the parts in order
rebuilds a single archive. Each labeled face is cropped and written to
``<out_root>/train/<class>/<image_stem>_<face_id>.jpg`` so ExpW plugs into
the same ``extra_train_sources`` machinery as RAF-DB / AffectNet.

All steps are idempotent: re-running skips completed stages, so Colab
sessions only pay the extract+crop cost on a fresh runtime. A native 7z
CLI is used when present; otherwise ``p7zip-full`` is installed via
apt-get (Colab). py7zr is the last-resort fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

logger = logging.getLogger("fer.prepare_expw")

LABEL_TO_CLASS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONCAT_BLOB = "_origin_full.7z"
EXTRACTED_DIRNAME = "_extracted"
META_FILENAME = "prep_meta.json"
JPEG_QUALITY = 95
# Unique face crops in the current Kaggle pack (91793 label rows, some dup keys).
MIN_EXPECTED_CROPS = 80_000


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _concat_volumes(raw_root: Path, dest_dir: Path) -> Path:
    """Concatenate ``origin.7z.*`` split volumes into a writable archive.

    Colab's kagglehub "Colab cache" is ``/kaggle/input/...`` and is
    read-only, so the blob must not be written next to the volumes.
    """
    parts = sorted(
        (p for p in raw_root.glob("origin.7z.*") if p.name != CONCAT_BLOB),
        key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)],
    )
    if not parts:
        raise FileNotFoundError(f"No origin.7z.* volumes under {raw_root}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    blob = dest_dir / CONCAT_BLOB
    expected = sum(p.stat().st_size for p in parts)
    if blob.exists() and blob.stat().st_size == expected:
        logger.info("Concatenated archive already present: %s", blob)
        return blob
    logger.info("Concatenating %d volumes → %s", len(parts), blob)
    with blob.open("wb") as out:
        for part in parts:
            with part.open("rb") as f:
                shutil.copyfileobj(f, out, length=1 << 22)
    if blob.stat().st_size != expected:
        blob.unlink()
        raise RuntimeError("Concatenated archive size mismatch; removed partial blob")
    return blob


def _find_7z_cli() -> str | None:
    return next((c for c in ("7zz", "7z", "7za") if shutil.which(c)), None)


def _ensure_7z_cli() -> str | None:
    """Return a 7z binary, installing p7zip-full via apt-get when possible.

    Colab runtimes are root and have apt-get; a Mac with Homebrew ``7zz``
    is left alone. ``sudo -n`` is used off-root so a password prompt never
    blocks training.
    """
    import subprocess

    existing = _find_7z_cli()
    if existing:
        return existing
    apt_get = shutil.which("apt-get")
    if not apt_get:
        return None
    install = [apt_get, "install", "-y", "p7zip-full"]
    euid = os.geteuid() if hasattr(os, "geteuid") else 1
    if euid != 0:
        sudo = shutil.which("sudo")
        if not sudo:
            return None
        install = [sudo, "-n", *install]
    logger.info("No 7z CLI on PATH — installing p7zip-full")
    try:
        subprocess.run(install, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Could not install p7zip-full: %s", exc)
        return None
    return _find_7z_cli()


def _extract_archive(blob: Path, dest_dir: Path) -> Path:
    """Extract the concatenated archive once; return the image directory.

    The pack is a *solid* LZMA archive (~4 GB single block), which py7zr
    decompresses pathologically slowly. Prefer a native 7z CLI (``7zz`` /
    ``7z`` / ``7za``; Colab gets ``p7zip-full`` via apt-get) and fall back
    to py7zr only when no CLI can be installed.
    """
    import subprocess

    extracted = dest_dir / EXTRACTED_DIRNAME
    image_dir = extracted / "origin"
    marker = extracted / ".extract_complete"
    if marker.exists() and image_dir.is_dir():
        logger.info("Archive already extracted: %s", image_dir)
        return image_dir
    logger.info("Extracting %s → %s (one-time, ~8 GB)", blob, extracted)
    extracted.mkdir(parents=True, exist_ok=True)
    cli = _ensure_7z_cli()
    if cli:
        subprocess.run([cli, "x", str(blob), f"-o{extracted}", "-y"], check=True)
    else:
        logger.warning(
            "No 7z CLI on PATH — falling back to py7zr (hours on this solid archive)"
        )
        import py7zr

        with py7zr.SevenZipFile(blob, "r") as z:
            z.extractall(path=extracted)
    marker.write_text("ok\n", encoding="utf-8")
    return image_dir


def default_prepared_root(raw_root: Path, name: str = "expw") -> Path:
    """Writable work dir for concat / extract / ImageFolder crops.

    Never write into ``raw_root``: on Colab, kagglehub may resolve to the
    read-only ``/kaggle/input/...`` mount (Errno 30). Default is
    ``~/.cache/fer-system/<name>`` on the VM disk. ``FER_PREPARED_ROOT``
    overrides the parent directory (``<override>/<name>``).
    """
    override = os.environ.get("FER_PREPARED_ROOT")
    if override:
        return Path(override) / name
    return Path.home() / ".cache" / "fer-system" / name


def _count_crops(train_root: Path) -> tuple[int, dict[str, int]]:
    """Count JPEG/PNG crops per class under ``train_root/<class>/``."""
    counts = {name: 0 for name in LABEL_TO_CLASS.values()}
    if not train_root.is_dir():
        return 0, counts
    total = 0
    for class_name in LABEL_TO_CLASS.values():
        class_dir = train_root / class_name
        if not class_dir.is_dir():
            continue
        n = sum(
            1
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[class_name] = n
        total += n
    return total, counts


def _parse_labels(label_path: Path) -> list[tuple[str, int, tuple[int, int, int, int], float, int]]:
    rows = []
    with label_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.split()
            if len(parts) != 8:
                continue
            name, face_id, top, left, right, bottom, conf, label = parts
            try:
                rows.append(
                    (
                        name,
                        int(face_id),
                        (int(left), int(top), int(right), int(bottom)),
                        float(conf),
                        int(label),
                    )
                )
            except ValueError:
                logger.warning("Skipping malformed label.lst line %d: %r", lineno, line[:80])
    return rows


def prepare_expw(
    raw_root: Path,
    out_root: Path,
    min_confidence: float = 0.0,
) -> Path:
    """Build ``<out_root>/train/<class>/`` from the raw ExpW pack; return out_root.

    Idempotent only when on-disk crop counts match ``prep_meta.json`` and
    every class is non-empty. A stale Drive tree with meta but 1k files
    will be rebuilt.
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    label_path = raw_root / "label.lst"
    if not label_path.is_file():
        raise FileNotFoundError(f"label.lst not found under {raw_root}")
    label_sha = _sha256_file(label_path)
    train_root = out_root / "train"
    disk_total, disk_counts = _count_crops(train_root)

    meta_path = out_root / META_FILENAME
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_n = int(meta.get("num_crops") or 0)
        complete = (
            meta.get("label_sha256") == label_sha
            and meta.get("min_confidence") == min_confidence
            and disk_total == meta_n
            and disk_total >= MIN_EXPECTED_CROPS
            and all(disk_counts[c] > 0 for c in LABEL_TO_CLASS.values())
        )
        if complete:
            logger.info(
                "ExpW already prepared at %s (%d crops, per-class=%s)",
                out_root,
                disk_total,
                disk_counts,
            )
            return out_root
        logger.warning(
            "ExpW prep incomplete or stale at %s (meta=%s disk=%d per-class=%s) — rebuilding",
            out_root,
            meta_n,
            disk_total,
            disk_counts,
        )

    blob = _concat_volumes(raw_root, out_root)
    image_dir = _extract_archive(blob, out_root)

    rows = _parse_labels(label_path)
    logger.info("label.lst: %d face annotations", len(rows))

    train_root = out_root / "train"
    for class_name in LABEL_TO_CLASS.values():
        (train_root / class_name).mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in LABEL_TO_CLASS.values()}
    skipped = {"confidence": 0, "bad_box": 0, "missing_image": 0, "unreadable": 0}
    num_crops = 0
    for i, (name, face_id, box, conf, label) in enumerate(rows):
        class_name = LABEL_TO_CLASS.get(label)
        if class_name is None:
            continue
        if conf < min_confidence:
            skipped["confidence"] += 1
            continue
        left, top, right, bottom = box
        if right <= left or bottom <= top:
            skipped["bad_box"] += 1
            continue
        src = image_dir / name
        if not src.is_file():
            skipped["missing_image"] += 1
            continue
        dst = train_root / class_name / f"{Path(name).stem}_{face_id}.jpg"
        if dst.exists():
            counts[class_name] += 1
            num_crops += 1
            continue
        try:
            with Image.open(src) as im:
                w, h = im.size
                crop = im.crop((max(0, left), max(0, top), min(w, right), min(h, bottom)))
                crop.convert("RGB").save(dst, "JPEG", quality=JPEG_QUALITY)
        except Exception as exc:  # corrupt image — skip, don't abort 90k crops
            skipped["unreadable"] += 1
            logger.debug("Unreadable %s: %s", src, exc)
            continue
        counts[class_name] += 1
        num_crops += 1
        if num_crops % 10000 == 0:
            logger.info("  cropped %d / %d faces", num_crops, len(rows))

    meta = {
        "source": "shahzadabbas/expression-in-the-wild-expw-dataset",
        "label_sha256": label_sha,
        "min_confidence": min_confidence,
        "num_crops": num_crops,
        "counts": counts,
        "skipped": skipped,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("ExpW prepared: %d crops → %s | counts=%s skipped=%s", num_crops, train_root, counts, skipped)
    empty = [c for c, n in counts.items() if n == 0]
    if empty or num_crops < MIN_EXPECTED_CROPS:
        raise RuntimeError(
            f"ExpW prepare produced {num_crops} crops (need ≥ {MIN_EXPECTED_CROPS}) "
            f"with empty classes {empty}"
        )
    return out_root
