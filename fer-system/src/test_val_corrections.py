"""METRIC: unsure rows are dropped; wrong official labels are remapped.

Run from fer-system/:

    python -m src.test_val_corrections
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset import FERDataset  # noqa: E402
from src.val_corrections import (  # noqa: E402
    apply_corrections,
    build_corrections,
    load_corrections,
    sample_key,
)

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _item(item_id: str, official: str, predicted: str, filename: str) -> dict:
    return {
        "id": item_id,
        "official_label": official,
        "predicted_label": predicted,
        "source_path": f"/cache/validation/{official}/{filename}",
    }


def test_build_drop_remap_keep() -> None:
    review = {
        "items": [
            _item("01", "angry", "fear", "a.jpg"),
            _item("02", "disgust", "angry", "b.jpg"),
            _item("03", "fear", "sad", "c.jpg"),
        ]
    }
    judgments = {"01": "wrong", "02": "unsure", "03": "correct"}
    spec = build_corrections(review, judgments)
    _assert(spec["n_drop"] == 1, spec)
    _assert(spec["n_remap"] == 1, spec)
    _assert(spec["n_keep"] == 1, spec)
    _assert(spec["drop"] == ["disgust/b.jpg"], spec["drop"])
    _assert(spec["remap"] == {"angry/a.jpg": "fear"}, spec["remap"])


def test_apply_on_imagefolder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = {
            "angry/a.jpg": "angry",
            "disgust/b.jpg": "disgust",
            "fear/c.jpg": "fear",
            "happy/keep.jpg": "happy",
        }
        for rel in files:
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (8, 8), color=40).save(dest)
        for name in CLASS_NAMES:
            (root / name).mkdir(exist_ok=True)

        ds = FERDataset(root, CLASS_NAMES, source_name="toy_val")
        _assert(len(ds) == 4, len(ds))
        spec = {
            "drop": ["disgust/b.jpg"],
            "remap": {"angry/a.jpg": "fear"},
        }
        stats = apply_corrections(ds, spec)
        _assert(stats["dropped"] == 1, stats)
        _assert(stats["remapped"] == 1, stats)
        _assert(len(ds) == 3, len(ds))
        by_key = {sample_key(path): label for path, label in ds.samples}
        _assert("disgust/b.jpg" not in by_key, by_key)
        _assert(by_key["angry/a.jpg"] == CLASS_NAMES.index("fear"), by_key)
        _assert(by_key["fear/c.jpg"] == CLASS_NAMES.index("fear"), by_key)


def test_committed_file_matches_audit_counts() -> None:
    spec = load_corrections("data/label_review/val_corrections.json")
    _assert(spec is not None, "committed corrections missing")
    _assert(spec["n_drop"] == 4, spec["n_drop"])
    _assert(spec["n_remap"] == 13, spec["n_remap"])
    _assert(spec["n_keep"] == 8, spec["n_keep"])
    _assert(len(spec["drop"]) == 4, spec["drop"])
    _assert(len(spec["remap"]) == 13, spec["remap"])
    from src.utils import load_config

    cfg = load_config("configs/public.yaml")
    names = [str(e.get("name") or "") for e in cfg["data"].get("extra_train_sources") or []]
    _assert("val_errors" not in names, names)
    _assert("human_feedback" not in names, names)
    # Reverted 2026-08-24: remapping val to the audit-time model prediction is
    # circular eval. The committed JSON is audit documentation only.
    _assert(
        cfg["data"].get("val_label_corrections") in (None, "", "null"),
        cfg["data"].get("val_label_corrections"),
    )


def main() -> None:
    test_build_drop_remap_keep()
    test_apply_on_imagefolder()
    test_committed_file_matches_audit_counts()
    print("METRIC ok: drop 4 unsure, remap 13 wrong, keep 8 correct; not extra_train")


if __name__ == "__main__":
    main()
