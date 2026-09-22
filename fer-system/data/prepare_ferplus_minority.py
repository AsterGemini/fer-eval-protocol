"""Prepare FER+ high-agreement disgust/fear crops as an ImageFolder tree.

FER+ (Barsoum et al., 2016) re-labels official FER2013 rows with 10 crowd
votes. This hook keeps **Training** rows only (never PublicTest/PrivateTest)
whose majority vote is disgust or fear with a clear agreement margin.

That is label quality, not another FER2013 val mirror: the same faces may
already appear in the primary train split, but with noisier single labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from src.utils import setup_logging

logger = logging.getLogger("fer.prepare_ferplus")

FERPLUS_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
FERPLUS_FILENAME = "fer2013new.csv"
META_FILENAME = "prep_meta.json"

# FER+ vote columns that map onto this project's 7 classes. contempt is dropped.
VOTE_TO_CLASS = {
    "neutral": "neutral",
    "happiness": "happy",
    "surprise": "surprise",
    "sadness": "sad",
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
}
TARGET_CLASSES = ("disgust", "fear")
ALL_CLASSES = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)
DEFAULT_MIN_VOTES = 6
DEFAULT_MAX_UNKNOWN_NF = 2
TRAIN_USAGE = "training"
# FER+ Training disgust with ≥6/10 votes is naturally small (~45).
MIN_EXPECTED_PER_TARGET = 40


def _norm_header(name: str) -> str:
    return name.strip().lstrip("\ufeff").lower().replace(" ", "_")


def find_fer2013_csv(raw_root: Path) -> Path:
    """Locate official-style ``fer2013.csv`` / ``icml_face_data.csv`` under a download root."""
    raw_root = Path(raw_root)
    names = ("fer2013.csv", "icml_face_data.csv")
    hits: list[Path] = []
    for name in names:
        hits.extend(p for p in raw_root.rglob(name) if p.is_file())
    if not hits:
        raise FileNotFoundError(
            f"No fer2013.csv / icml_face_data.csv under {raw_root}. "
            "Need the official FER2013 CSV (emotion, pixels, Usage), not an ImageFolder pack."
        )
    hits.sort(key=lambda p: (0 if p.name.lower() == "fer2013.csv" else 1, len(p.parts)))
    return hits[0]


def download_ferplus_csv(dest: Path) -> Path:
    """Fetch ``fer2013new.csv`` from Microsoft/FERPlus if it is not already on disk."""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 100_000:
        logger.info("FER+ labels already present: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading FER+ labels → %s", dest)
    req = urllib.request.Request(
        FERPLUS_URL,
        headers={"User-Agent": "fer-system/ferplus-minority"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
        out.write(resp.read())
    if dest.stat().st_size < 100_000:
        raise RuntimeError(f"FER+ download looks truncated: {dest} ({dest.stat().st_size} bytes)")
    return dest


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        field_map = {name: _norm_header(name) for name in reader.fieldnames}
        rows: list[dict[str, str]] = []
        for raw in reader:
            rows.append({field_map[k]: (v or "").strip() if v is not None else "" for k, v in raw.items()})
        return rows


def _int_vote(row: dict[str, str], key: str) -> int:
    raw = row.get(key, "0") or "0"
    return int(raw)


def majority_minority_label(
    votes: dict[str, int],
    *,
    min_votes: int = DEFAULT_MIN_VOTES,
    max_unknown_nf: int = DEFAULT_MAX_UNKNOWN_NF,
    targets: Iterable[str] = TARGET_CLASSES,
) -> str | None:
    """Return disgust/fear when crowd agreement is clear; else None.

    ``votes`` keys are FER+ column names (happiness, anger, unknown, NF, ...).
    Ties, weak majorities, and high unknown/NF are rejected.
    """
    target_set = set(targets)
    unknown_nf = int(votes.get("unknown", 0)) + int(votes.get("NF", 0))
    if unknown_nf > max_unknown_nf:
        return None
    ranked = sorted(
        ((VOTE_TO_CLASS[name], int(votes[name])) for name in VOTE_TO_CLASS),
        key=lambda item: (-item[1], item[0]),
    )
    top_name, top_votes = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if top_votes < min_votes or top_votes == runner_up:
        return None
    if top_name not in target_set:
        return None
    return top_name


def _pixels_to_image(pixel_str: str) -> Image.Image:
    values = [int(p) for p in pixel_str.split()]
    if len(values) != 48 * 48:
        raise ValueError(f"Expected 2304 pixels, got {len(values)}")
    return Image.frombytes("L", (48, 48), bytes(values))


def select_minority_rows(
    fer_rows: list[dict[str, str]],
    plus_rows: list[dict[str, str]],
    *,
    min_votes: int = DEFAULT_MIN_VOTES,
    max_unknown_nf: int = DEFAULT_MAX_UNKNOWN_NF,
) -> list[dict[str, Any]]:
    """Align official FER2013 rows with FER+ votes; keep Training minority crops."""
    if len(fer_rows) != len(plus_rows):
        raise ValueError(
            f"FER2013 ({len(fer_rows)} rows) and FER+ ({len(plus_rows)} rows) "
            "must be the same length and order"
        )
    selected: list[dict[str, Any]] = []
    for idx, (fer, plus) in enumerate(zip(fer_rows, plus_rows)):
        usage = (plus.get("usage") or fer.get("usage") or "").strip().lower()
        if usage != TRAIN_USAGE:
            continue
        votes = {
            key: _int_vote(plus, key)
            for key in (*VOTE_TO_CLASS, "contempt", "unknown", "nf")
        }
        # DictReader normalizes NF → nf
        votes["NF"] = votes.pop("nf", 0)
        label = majority_minority_label(
            votes, min_votes=min_votes, max_unknown_nf=max_unknown_nf
        )
        if label is None:
            continue
        pixels = fer.get("pixels") or ""
        image_name = plus.get("image_name") or f"fer{idx:07d}.png"
        selected.append(
            {
                "index": idx,
                "image_name": image_name,
                "label": label,
                "pixels": pixels,
                "votes": top_vote_summary(votes),
            }
        )
    return selected


def top_vote_summary(votes: dict[str, int]) -> dict[str, int]:
    """Keep the emotion + quality votes for audit metadata."""
    keep = list(VOTE_TO_CLASS) + ["contempt", "unknown", "NF"]
    return {k: int(votes.get(k, 0)) for k in keep}


def write_imagefolder(
    rows: list[dict[str, Any]],
    train_root: Path,
) -> dict[str, int]:
    """Write selected crops under ``train_root/<class>/``; return per-class counts."""
    train_root = Path(train_root)
    for name in ALL_CLASSES:
        (train_root / name).mkdir(parents=True, exist_ok=True)
    counts = {name: 0 for name in ALL_CLASSES}
    for row in rows:
        label = str(row["label"])
        stem = Path(str(row["image_name"])).stem
        dest = train_root / label / f"{stem}.png"
        image = _pixels_to_image(str(row["pixels"]))
        image.save(dest, format="PNG")
        counts[label] += 1
    return counts


def prepare_ferplus_minority(
    raw_root: Path,
    out_root: Path,
    *,
    min_votes: int = DEFAULT_MIN_VOTES,
    max_unknown_nf: int = DEFAULT_MAX_UNKNOWN_NF,
) -> Path:
    """Build ``<out_root>/train/<class>/`` from FER2013 CSV + FER+ votes."""
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    fer_csv = find_fer2013_csv(raw_root)
    plus_csv = download_ferplus_csv(out_root / FERPLUS_FILENAME)
    train_root = out_root / "train"
    meta_path = out_root / META_FILENAME

    params = {
        "min_votes": int(min_votes),
        "max_unknown_nf": int(max_unknown_nf),
        "targets": list(TARGET_CLASSES),
        "fer_csv": str(fer_csv),
        "plus_csv": str(plus_csv),
        "plus_bytes": plus_csv.stat().st_size,
        "fer_bytes": fer_csv.stat().st_size,
    }
    if meta_path.is_file() and train_root.is_dir():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        disk_counts = {
            name: sum(1 for p in (train_root / name).glob("*.png"))
            if (train_root / name).is_dir()
            else 0
            for name in ALL_CLASSES
        }
        if (
            meta.get("params") == params
            and meta.get("counts") == disk_counts
            and all(disk_counts[c] >= MIN_EXPECTED_PER_TARGET for c in TARGET_CLASSES)
        ):
            logger.info(
                "FER+ minority already prepared at %s (counts=%s)",
                out_root,
                disk_counts,
            )
            return out_root
        logger.warning("FER+ minority prep stale at %s — rebuilding", out_root)

    fer_rows = _read_csv_dicts(fer_csv)
    plus_rows = _read_csv_dicts(plus_csv)
    selected = select_minority_rows(
        fer_rows,
        plus_rows,
        min_votes=min_votes,
        max_unknown_nf=max_unknown_nf,
    )
    if train_root.exists():
        import shutil

        shutil.rmtree(train_root)
    counts = write_imagefolder(selected, train_root)
    meta = {
        "source": "FER+ Training minority (disgust/fear)",
        "ferplus_url": FERPLUS_URL,
        "params": params,
        "num_crops": sum(counts.values()),
        "counts": counts,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("FER+ minority prepared: %d crops → %s | counts=%s", sum(counts.values()), train_root, counts)
    missing = [c for c in TARGET_CLASSES if counts[c] < MIN_EXPECTED_PER_TARGET]
    if missing:
        raise RuntimeError(
            f"FER+ minority produced too few crops for {missing}: {counts}. "
            f"Need ≥ {MIN_EXPECTED_PER_TARGET} each."
        )
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FER+ high-agreement disgust/fear crops")
    parser.add_argument("--raw-root", type=str, required=True, help="Kaggle download root that contains fer2013.csv")
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--min-votes", type=int, default=DEFAULT_MIN_VOTES)
    parser.add_argument("--max-unknown-nf", type=int, default=DEFAULT_MAX_UNKNOWN_NF)
    args = parser.parse_args()
    setup_logging("fer.prepare_ferplus")
    prepare_ferplus_minority(
        Path(args.raw_root),
        Path(args.out_root),
        min_votes=args.min_votes,
        max_unknown_nf=args.max_unknown_nf,
    )


if __name__ == "__main__":
    main()
