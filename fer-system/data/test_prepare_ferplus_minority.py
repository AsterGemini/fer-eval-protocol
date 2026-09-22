"""METRIC check: FER+ minority keep/reject rules.

Run from fer-system/:

    python -m data.test_prepare_ferplus_minority
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.prepare_ferplus_minority import (  # noqa: E402
    find_fer2013_csv,
    majority_minority_label,
    select_minority_rows,
    write_imagefolder,
)

PIXELS = " ".join(["0"] * (48 * 48))


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_majority_keeps_clear_fear() -> None:
    votes = {
        "neutral": 0,
        "happiness": 0,
        "surprise": 1,
        "sadness": 0,
        "anger": 0,
        "disgust": 0,
        "fear": 8,
        "unknown": 1,
        "NF": 0,
    }
    _assert(majority_minority_label(votes) == "fear", votes)


def test_majority_keeps_clear_disgust() -> None:
    votes = {
        "neutral": 0,
        "happiness": 0,
        "surprise": 0,
        "sadness": 1,
        "anger": 1,
        "disgust": 7,
        "fear": 0,
        "unknown": 1,
        "NF": 0,
    }
    _assert(majority_minority_label(votes) == "disgust", votes)


def test_majority_rejects_happy_and_ties_and_noise() -> None:
    happy = {
        "neutral": 0,
        "happiness": 10,
        "surprise": 0,
        "sadness": 0,
        "anger": 0,
        "disgust": 0,
        "fear": 0,
        "unknown": 0,
        "NF": 0,
    }
    _assert(majority_minority_label(happy) is None, "happy is not a minority target")
    tie = {
        "neutral": 0,
        "happiness": 0,
        "surprise": 0,
        "sadness": 0,
        "anger": 0,
        "disgust": 5,
        "fear": 5,
        "unknown": 0,
        "NF": 0,
    }
    _assert(majority_minority_label(tie) is None, "tie must be rejected")
    weak = {
        "neutral": 2,
        "happiness": 0,
        "surprise": 0,
        "sadness": 1,
        "anger": 0,
        "disgust": 4,
        "fear": 1,
        "unknown": 2,
        "NF": 0,
    }
    _assert(majority_minority_label(weak) is None, "4 votes is below the default floor")
    noisy = {
        "neutral": 0,
        "happiness": 0,
        "surprise": 0,
        "sadness": 0,
        "anger": 0,
        "disgust": 7,
        "fear": 0,
        "unknown": 2,
        "NF": 1,
    }
    _assert(majority_minority_label(noisy) is None, "unknown+NF=3 exceeds default cap")


def test_select_keeps_training_only() -> None:
    fer_rows = [
        {"emotion": "2", "pixels": PIXELS, "usage": "Training"},
        {"emotion": "1", "pixels": PIXELS, "usage": "PublicTest"},
        {"emotion": "2", "pixels": PIXELS, "usage": "Training"},
    ]
    plus_rows = [
        {
            "usage": "Training",
            "image_name": "fer0000000.png",
            "neutral": "0",
            "happiness": "0",
            "surprise": "0",
            "sadness": "0",
            "anger": "0",
            "disgust": "0",
            "fear": "9",
            "contempt": "0",
            "unknown": "1",
            "nf": "0",
        },
        {
            "usage": "PublicTest",
            "image_name": "fer0000001.png",
            "neutral": "0",
            "happiness": "0",
            "surprise": "0",
            "sadness": "0",
            "anger": "0",
            "disgust": "9",
            "fear": "0",
            "contempt": "0",
            "unknown": "1",
            "nf": "0",
        },
        {
            "usage": "Training",
            "image_name": "fer0000002.png",
            "neutral": "0",
            "happiness": "10",
            "surprise": "0",
            "sadness": "0",
            "anger": "0",
            "disgust": "0",
            "fear": "0",
            "contempt": "0",
            "unknown": "0",
            "nf": "0",
        },
    ]
    selected = select_minority_rows(fer_rows, plus_rows)
    _assert(len(selected) == 1, selected)
    _assert(selected[0]["label"] == "fear", selected[0])
    _assert(selected[0]["image_name"] == "fer0000000.png", selected[0])


def test_write_imagefolder_only_populates_targets() -> None:
    rows = [
        {"image_name": "fer0000000.png", "label": "fear", "pixels": PIXELS},
        {"image_name": "fer0000001.png", "label": "disgust", "pixels": PIXELS},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "train"
        counts = write_imagefolder(rows, root)
        _assert(counts["fear"] == 1, counts)
        _assert(counts["disgust"] == 1, counts)
        _assert(counts["happy"] == 0, counts)
        fear = Image.open(root / "fear" / "fer0000000.png")
        _assert(fear.size == (48, 48), fear.size)
        _assert((root / "happy").is_dir(), "all 7 class folders must exist")


def test_find_fer2013_csv_prefers_official_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nested = root / "inner"
        nested.mkdir()
        (nested / "icml_face_data.csv").write_text("emotion,pixels,Usage\n", encoding="utf-8")
        official = root / "fer2013.csv"
        official.write_text("emotion,pixels,Usage\n", encoding="utf-8")
        found = find_fer2013_csv(root)
        _assert(found == official, found)


def test_select_rejects_length_mismatch() -> None:
    try:
        select_minority_rows([{"pixels": PIXELS, "usage": "Training"}], [])
    except ValueError as exc:
        _assert("same length" in str(exc), str(exc))
    else:
        raise AssertionError("expected ValueError on length mismatch")


def _csv_roundtrip_headers() -> None:
    """Sanity: DictReader + header normalize matches prepare_ferplus._read_csv_dicts."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fer2013.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["emotion", "pixels", "Usage"])
            writer.writerow(["2", PIXELS, "Training"])
        from data.prepare_ferplus_minority import _read_csv_dicts

        rows = _read_csv_dicts(path)
        _assert(rows[0]["usage"] == "Training", rows[0])
        _assert(rows[0]["pixels"] == PIXELS, "pixels should survive")


def main() -> None:
    test_majority_keeps_clear_fear()
    test_majority_keeps_clear_disgust()
    test_majority_rejects_happy_and_ties_and_noise()
    test_select_keeps_training_only()
    test_write_imagefolder_only_populates_targets()
    test_find_fer2013_csv_prefers_official_name()
    test_select_rejects_length_mismatch()
    _csv_roundtrip_headers()
    print(
        "METRIC ok: FER+ keep=Training disgust/fear with ≥6 votes; "
        "reject ties, PublicTest, happy, and high unknown/NF"
    )


if __name__ == "__main__":
    main()
