"""METRIC check: human label → eval-only crop + JSONL; skip writes no image.

Run from fer-system/:

    python app/test_feedback.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _APP_DIR]
sys.path.insert(0, str(_ROOT))

from app.feedback import (  # noqa: E402
    EVAL_ONLY_NOTICE,
    SKIP_LABEL,
    eval_dir,
    jsonl_path,
    record_feedback,
)
from src.utils import load_config  # noqa: E402

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def _crop() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(40, 80, 120))


def _result(label: str = "neutral") -> dict:
    probs = {name: (0.7 if name == label else 0.05) for name in CLASS_NAMES}
    return {
        "predicted_label": label,
        "confidence": 0.7,
        "all_probabilities": probs,
    }


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_eval_only_notice() -> None:
    text = EVAL_ONLY_NOTICE.lower()
    _assert("eval-only" in text, "UI notice must say eval-only")
    _assert(
        "training set" in text or "not added to the training" in text,
        "UI notice must say the crop is not used for training",
    )
    _assert("face crop" in text, "UI notice must refer to the cropped picture")


def test_labeled_submit_writes_eval_crop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        status = record_feedback(
            _crop(),
            _result("happy"),
            "happy",
            CLASS_NAMES,
            project_dir=root,
        )
        _assert(status["ok"], status["message"])
        _assert("eval-only" in status["message"].lower(), status["message"])
        _assert("not in the training set" in status["message"].lower(), status["message"])
        row = status["row"]
        _assert(row["eval_only"] is True, "row must be marked eval_only")
        _assert(row["skipped"] is False, "labeled row must not be skipped")
        _assert(row["human_label"] == "happy", row)
        crop_file = eval_dir(root) / "happy" / f"{row['id']}.jpg"
        _assert(crop_file.is_file(), f"missing eval crop {crop_file}")
        lines = jsonl_path(root).read_text(encoding="utf-8").strip().splitlines()
        _assert(len(lines) == 1, f"expected 1 jsonl line, got {len(lines)}")
        parsed = json.loads(lines[0])
        _assert(parsed["eval_only"] is True, parsed)
        _assert(parsed["crop_path"] is not None, parsed)
        # No other class folders should have files.
        for name in CLASS_NAMES:
            if name == "happy":
                continue
            other = eval_dir(root) / name
            _assert(
                not other.exists() or not any(other.iterdir()),
                f"unexpected files under {other}",
            )


def test_skip_writes_no_eval_image() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        status = record_feedback(
            _crop(),
            _result(),
            SKIP_LABEL,
            CLASS_NAMES,
            project_dir=root,
        )
        _assert(status["ok"], status["message"])
        _assert(status["row"]["skipped"] is True, status["row"])
        _assert(status["row"]["crop_path"] is None, status["row"])
        eval_root = eval_dir(root)
        jpgs = list(eval_root.rglob("*.jpg")) if eval_root.exists() else []
        _assert(jpgs == [], f"skip must not write eval crops, got {jpgs}")
        lines = jsonl_path(root).read_text(encoding="utf-8").strip().splitlines()
        _assert(len(lines) == 1, lines)
        _assert(json.loads(lines[0])["skipped"] is True, lines[0])


def test_no_crop_and_no_label_are_noops() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        missing = record_feedback(None, None, "happy", CLASS_NAMES, project_dir=root)
        _assert(not missing["ok"], missing)
        unlabeled = record_feedback(
            _crop(), _result(), None, CLASS_NAMES, project_dir=root
        )
        _assert(not unlabeled["ok"], unlabeled)
        _assert(not jsonl_path(root).exists(), "failed submits must not write jsonl")
        _assert(
            not eval_dir(root).exists() or not any(eval_dir(root).rglob("*.jpg")),
            "failed submits must not write crops",
        )


def test_training_config_untouched() -> None:
    cfg = load_config("configs/public.yaml")
    extras = cfg.get("data", {}).get("extra_train_sources") or []
    names = [str(e.get("name") or "") for e in extras]
    _assert(
        "human_feedback" not in names,
        f"v1 must not wire human_feedback into extra_train_sources: {names}",
    )
    app_src = (_ROOT / "app" / "app.py").read_text(encoding="utf-8")
    _assert("EVAL_ONLY_NOTICE" in app_src, "app.py must show EVAL_ONLY_NOTICE")
    _assert("record_feedback" in app_src, "app.py must call record_feedback")
    _assert("Save label (eval-only)" in app_src, "submit button must say eval-only")


def main() -> None:
    test_eval_only_notice()
    test_labeled_submit_writes_eval_crop()
    test_skip_writes_no_eval_image()
    test_no_crop_and_no_label_are_noops()
    test_training_config_untouched()
    print(
        "METRIC ok: labeled submit → 1 eval crop + jsonl; "
        "skip/no-face → no eval image; extra_train_sources unchanged"
    )


if __name__ == "__main__":
    main()
