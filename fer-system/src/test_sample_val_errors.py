"""METRIC check: val-error sample prefers minority official labels.

Run from fer-system/:

    python -m src.test_sample_val_errors
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.sample_val_errors import collect_errors, sample_errors, write_review_pack  # noqa: E402

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _err(official: str, pred: str, conf: float, path: str) -> dict:
    return {
        "path": path,
        "official_label": official,
        "predicted_label": pred,
        "confidence": conf,
        "official_probability": 1.0 - conf,
        "all_probabilities": {name: 0.0 for name in CLASS_NAMES},
    }


def test_collect_errors_drops_matches() -> None:
    rows = [
        _err("fear", "fear", 0.9, "a.png"),
        _err("fear", "sad", 0.8, "b.png"),
    ]
    # first row is not an error — official == predicted
    rows[0]["predicted_label"] = "fear"
    rows[0]["official_label"] = "fear"
    kept = collect_errors(rows)
    _assert(len(kept) == 1, kept)
    _assert(kept[0]["path"] == "b.png", kept)


def test_sample_fills_minority_quota_and_is_deterministic() -> None:
    errors = []
    for i in range(20):
        errors.append(_err("disgust", "sad", 0.9 - i * 0.01, f"d{i}.png"))
    for i in range(20):
        errors.append(_err("fear", "sad", 0.95 - i * 0.01, f"f{i}.png"))
    for i in range(20):
        errors.append(_err("happy", "neutral", 0.8 - i * 0.01, f"h{i}.png"))
    a = sample_errors(errors, 25, seed=42)
    b = sample_errors(errors, 25, seed=42)
    _assert([r["path"] for r in a] == [r["path"] for r in b], "seed 42 must be stable")
    counts: dict[str, int] = {}
    for row in a:
        counts[row["official_label"]] = counts.get(row["official_label"], 0) + 1
    _assert(counts.get("disgust", 0) >= 8, counts)
    _assert(counts.get("fear", 0) >= 8, counts)
    _assert(len(a) == 25, len(a))
    disgust = [r for r in a if r["official_label"] == "disgust"]
    _assert(disgust[0]["path"] == "d0.png", "highest-confidence disgust error first in pool")


def test_write_review_pack_copies_images() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src.png"
        Image.new("RGB", (48, 48), color=(10, 20, 30)).save(src)
        sampled = [
            {
                "path": str(src),
                "official_label": "fear",
                "predicted_label": "sad",
                "confidence": 0.77,
                "official_probability": 0.1,
                "all_probabilities": {name: (0.77 if name == "sad" else 0.03) for name in CLASS_NAMES},
            }
        ]
        out = Path(tmp) / "review"
        payload = write_review_pack(sampled, out, CLASS_NAMES)
        _assert(payload["n"] == 1, payload)
        _assert((out / "index.html").is_file(), "missing index.html")
        _assert((out / "review.json").is_file(), "missing review.json")
        _assert((out / "images" / "01_fear_pred-sad.png").is_file(), "missing copied face")
        _assert(payload["items"][0]["human_verdict"] is None, "verdict starts empty")


def main() -> None:
    test_collect_errors_drops_matches()
    test_sample_fills_minority_quota_and_is_deterministic()
    test_write_review_pack_copies_images()
    print("METRIC ok: 25-error sample is seeded, minority-heavy, and writes a review pack")


if __name__ == "__main__":
    main()
