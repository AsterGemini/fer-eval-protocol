"""Human labels for Gradio crops — eval-only, never training.

Labeled crops land under ``data/human_feedback/eval/<class>/``.
JSONL is the audit log. Training configs must not point here (v1).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils import project_root

logger = logging.getLogger("fer.feedback")

SKIP_LABEL = "skip"

EVAL_ONLY_NOTICE = (
    "If you submit a label, the **face crop** (not the original photo) is saved "
    "as **eval-only**. It is not added to the training set and will not be used "
    "to update the model."
)


def feedback_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "data" / "human_feedback"


def eval_dir(root: Path | None = None) -> Path:
    return feedback_root(root) / "eval"


def jsonl_path(root: Path | None = None) -> Path:
    return feedback_root(root) / "feedback.jsonl"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def record_feedback(
    crop: Image.Image | None,
    result: dict[str, Any] | None,
    human_label: str | None,
    class_names: list[str],
    *,
    project_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist one human decision. Labeled crops go to the eval folder only.

    Returns a status dict with ``ok`` (bool) and ``message`` (str) for the UI.
    """
    names = list(class_names)
    label = (human_label or "").strip().lower()
    out_eval = eval_dir(project_dir)
    out_jsonl = jsonl_path(project_dir)

    if crop is None or not result:
        return {
            "ok": False,
            "message": "No face crop to save. Upload a photo and predict first.",
            "row": None,
        }
    if not label:
        return {
            "ok": False,
            "message": "Pick an emotion, or Skip to discard this crop.",
            "row": None,
        }
    if label != SKIP_LABEL and label not in names:
        return {
            "ok": False,
            "message": f"Unknown label {label!r}. Expected one of {names} or skip.",
            "row": None,
        }

    record_id = uuid.uuid4().hex
    row: dict[str, Any] = {
        "id": record_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "eval_only": True,
        "skipped": label == SKIP_LABEL,
        "human_label": None if label == SKIP_LABEL else label,
        "predicted_label": result.get("predicted_label"),
        "confidence": result.get("confidence"),
        "all_probabilities": result.get("all_probabilities"),
        "crop_path": None,
    }

    if label == SKIP_LABEL:
        _append_jsonl(out_jsonl, row)
        logger.info("Feedback skipped id=%s", record_id)
        return {
            "ok": True,
            "message": "Skipped. Nothing was added to the eval set.",
            "row": row,
        }

    class_dir = out_eval / label
    class_dir.mkdir(parents=True, exist_ok=True)
    crop_file = class_dir / f"{record_id}.jpg"
    crop.convert("RGB").save(crop_file, format="JPEG", quality=95)
    try:
        row["crop_path"] = str(crop_file.relative_to(project_root() if project_dir is None else project_dir))
    except ValueError:
        row["crop_path"] = str(crop_file)
    _append_jsonl(out_jsonl, row)
    logger.info("Eval-only crop saved id=%s label=%s path=%s", record_id, label, crop_file)
    return {
        "ok": True,
        "message": (
            f"Saved as eval-only ({label}) at {row['crop_path']}. "
            "This crop is not in the training set."
        ),
        "row": row,
    }
