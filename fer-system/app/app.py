"""Gradio web demo for facial emotion recognition.

Launch from fer-system/:

    python app/app.py
    python app/app.py --checkpoint /path/to/local.pt
    python app/app.py --ensemble

Default is ``paths.best_checkpoint`` in ``configs/public.yaml``
(``artifacts/models/efficientnet_b0/checkpoints/best.pt``). This repo does
not ship trained weights; missing files fail with a bring-your-own hint.
``--ensemble`` loads ``inference.ensemble.members`` without flipping
``enabled`` in public.yaml. Override with ``--checkpoint``,
FER_CHECKPOINT=/path/to/weights.pt, or
FER_CHECKPOINTS=/path/to/b0.pt,/path/to/b2.pt.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = Path(__file__).resolve().parent
# `python app/app.py` puts app/ on sys.path, which shadows the `app` package.
sys.path = [p for p in sys.path if Path(p).resolve() != _APP_DIR]
sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402

from app.feedback import (  # noqa: E402
    EVAL_ONLY_NOTICE,
    SKIP_LABEL,
    record_feedback,
)
from src.inference import load_predictor, predict_pil  # noqa: E402
from src.utils import load_config, project_root, setup_logging  # noqa: E402

logger = logging.getLogger("fer.app")

FACE_MARGIN = 0.2
NO_FACE_LABEL = "no face found"
BYO_CHECKPOINT_HINT = (
    "This repo does not redistribute trained weights. Train with "
    "`python -m src.train --model b0` (writes "
    "artifacts/models/efficientnet_b0/checkpoints/best.pt) or point "
    "--checkpoint / FER_CHECKPOINT at a local .pt you bring."
)


def checkpoint_path(path: str) -> Path:
    """Resolve a checkpoint path the same way inference does."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else project_root() / resolved


def checkpoint_is_file(path: str) -> bool:
    return checkpoint_path(path).is_file()


def load_face_cascade() -> cv2.CascadeClassifier:
    """Load OpenCV's bundled frontal-face Haar cascade (no extra download)."""
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {path}")
    return cascade


def detect_and_crop_face(
    image: Image.Image,
    cascade: cv2.CascadeClassifier,
    margin: float = FACE_MARGIN,
) -> Image.Image | None:
    """Return the largest detected face with margin, or None if none found."""
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, int(x) - mx)
    y1 = max(0, int(y) - my)
    x2 = min(rgb.shape[1], int(x) + int(w) + mx)
    y2 = min(rgb.shape[0], int(y) + int(h) + my)
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return Image.fromarray(crop)


def _ensemble_members(cfg: dict) -> tuple[list[str], list[float]]:
    ensemble_cfg = (cfg.get("inference") or {}).get("ensemble") or {}
    method = ensemble_cfg.get("method", "weighted_mean_probabilities")
    if method != "weighted_mean_probabilities":
        raise ValueError(
            f"Unsupported inference.ensemble.method={method!r}; "
            "use 'weighted_mean_probabilities'"
        )
    members = ensemble_cfg.get("members") or []
    if len(members) < 2:
        raise ValueError("An enabled inference ensemble requires at least two members")
    checkpoints = [str(member["checkpoint"]) for member in members]
    weights = [float(member.get("weight", 1.0)) for member in members]
    return checkpoints, weights


def resolve_predictor_spec(
    cfg: dict,
    *,
    ensemble: bool = False,
    extra_checkpoints: list[str] | None = None,
) -> tuple[list[str], list[float] | None]:
    """Resolve checkpoint paths and optional weights from env, CLI, or config.

    Precedence: FER_CHECKPOINTS, FER_CHECKPOINT, ``--checkpoint``, then yaml
    (``--ensemble`` / ``inference.ensemble.enabled`` members, else
    ``paths.best_checkpoint``). Configured ensembles remain opt-in so a
    missing future member cannot break today's single-model app.
    ``ensemble=True`` (CLI ``--ensemble``) uses yaml members without requiring
    ``inference.ensemble.enabled``.
    """
    env_checkpoints = os.environ.get("FER_CHECKPOINTS")
    if env_checkpoints:
        checkpoints = [
            value.strip() for value in env_checkpoints.split(",") if value.strip()
        ]
        if not checkpoints:
            raise ValueError("FER_CHECKPOINTS did not contain any checkpoint paths")
        env_weights = os.environ.get("FER_ENSEMBLE_WEIGHTS")
        weights = (
            [float(value.strip()) for value in env_weights.split(",")]
            if env_weights
            else None
        )
        return checkpoints, weights

    env_checkpoint = os.environ.get("FER_CHECKPOINT")
    if env_checkpoint:
        return [env_checkpoint], None

    if extra_checkpoints:
        checkpoints = [value.strip() for value in extra_checkpoints if value.strip()]
        if not checkpoints:
            raise ValueError("--checkpoint did not contain any checkpoint paths")
        return checkpoints, None

    ensemble_cfg = (cfg.get("inference") or {}).get("ensemble") or {}
    if ensemble or bool(ensemble_cfg.get("enabled", False)):
        return _ensemble_members(cfg)

    return [cfg["paths"]["best_checkpoint"]], None


def missing_checkpoint_message(missing: list[str]) -> str:
    joined = ", ".join(missing)
    return f"Missing checkpoint(s): {joined}. {BYO_CHECKPOINT_HINT}"


def select_existing_predictor_spec(
    cfg: dict,
    *,
    ensemble: bool = False,
    extra_checkpoints: list[str] | None = None,
    is_file=checkpoint_is_file,
) -> tuple[list[str], list[float] | None]:
    """Require every resolved checkpoint to exist on disk.

    Yaml B0, ``--checkpoint``, env, and ``--ensemble`` lists all fail with
    the same bring-your-own hint when a path is missing. Nothing is
    substituted from git.
    """
    checkpoints, weights = resolve_predictor_spec(
        cfg, ensemble=ensemble, extra_checkpoints=extra_checkpoints
    )
    missing = [path for path in checkpoints if not is_file(path)]
    if missing:
        raise SystemExit(missing_checkpoint_message(missing))
    return checkpoints, weights


def _demo_description(checkpoints: list[str]) -> str:
    loaded = ", ".join(checkpoints)
    description = (
        "Upload a photo. The app finds the largest face, crops it, and predicts "
        "one of 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise). "
        "If no face is found you get a warning instead of a guess. "
        f"This session loaded {loaded}. "
        "This is a research visualization, not a production emotion detector."
    )
    if len(checkpoints) >= 2:
        description += (
            f" Yaml B0+B2 ensemble ({len(checkpoints)} checkpoints)."
        )
    elif any("efficientnet_b0" in path.replace("\\", "/") for path in checkpoints):
        description += (
            " Held-out RAF-DB val macro F1 is 81.71% for this B0 selection "
            "checkpoint; FER2013-style val is ~0.39 (label noise)."
        )
    return description


def resolve_cascade_spec(cfg: dict) -> tuple[bool, float | None]:
    """Resolve entropy-cascade knobs from env or config.

    ``FER_CASCADE_MAX_ENTROPY`` forces cascade on (nats). Otherwise the nested
    ``inference.ensemble.cascade`` block is used. Callers with a single
    checkpoint should ignore the result.
    """
    env_max = os.environ.get("FER_CASCADE_MAX_ENTROPY")
    if env_max is not None and env_max.strip() != "":
        return True, float(env_max.strip())

    cascade_cfg = ((cfg.get("inference") or {}).get("ensemble") or {}).get("cascade") or {}
    enabled = bool(cascade_cfg.get("enabled", False))
    rule = str(cascade_cfg.get("rule", "entropy"))
    if enabled and rule != "entropy":
        raise ValueError(
            f"Unsupported inference.ensemble.cascade.rule={rule!r}; use 'entropy'"
        )
    if not enabled:
        return False, None
    max_entropy = cascade_cfg.get("max_entropy")
    return True, 0.8 if max_entropy is None else float(max_entropy)


def build_demo(
    *,
    ensemble: bool = False,
    extra_checkpoints: list[str] | None = None,
) -> gr.Blocks:
    cfg = load_config("configs/public.yaml")
    setup_logging("fer.app", log_dir=cfg["paths"]["log_dir"])
    checkpoints, weights = select_existing_predictor_spec(
        cfg, ensemble=ensemble, extra_checkpoints=extra_checkpoints
    )
    cascade_enabled, max_entropy = resolve_cascade_spec(cfg)
    if len(checkpoints) < 2:
        cascade_enabled = False
    logger.info("Loading predictor from %s", checkpoints)
    model, device, transform = load_predictor(
        cfg,
        checkpoints,
        weights=weights,
        cascade_enabled=cascade_enabled,
        max_entropy=max_entropy,
    )
    cascade = load_face_cascade()
    logger.info("Face cascade loaded; ready for requests")

    class_names = list(cfg["class_names"])
    empty: dict[str, float] = {}

    def predict(image: Image.Image | None):
        if image is None:
            return empty, None, None, None
        crop = detect_and_crop_face(image, cascade)
        if crop is None:
            return {NO_FACE_LABEL: 1.0}, None, None, None
        result = predict_pil(crop, model, device, transform, cfg, top_k=3)
        return result["all_probabilities"], crop, crop, result

    def submit_label(
        human_label: str | None,
        crop: Image.Image | None,
        result: dict | None,
    ) -> str:
        status = record_feedback(crop, result, human_label, class_names)
        return status["message"]

    description = _demo_description(checkpoints)
    label_choices = [(name, name) for name in class_names] + [
        ("Skip — do not save", SKIP_LABEL)
    ]

    with gr.Blocks(title="FER — Facial Emotion Recognition") as demo:
        last_crop = gr.State(None)
        last_result = gr.State(None)
        gr.Markdown("# Facial Emotion Recognition")
        gr.Markdown(description)
        with gr.Row():
            inp = gr.Image(type="pil", label="Photo")
            with gr.Column():
                labels = gr.Label(num_top_classes=3, label="Emotion")
                crop_out = gr.Image(type="pil", label="Face crop the model saw")
        gr.Markdown(EVAL_ONLY_NOTICE)
        human_label = gr.Radio(
            choices=label_choices,
            value=None,
            label="Your label for this crop",
        )
        save_btn = gr.Button("Save label (eval-only)")
        status = gr.Textbox(label="Feedback", interactive=False)
        predict_outputs = [labels, crop_out, last_crop, last_result]
        inp.change(fn=predict, inputs=inp, outputs=predict_outputs)
        gr.Button("Predict").click(fn=predict, inputs=inp, outputs=predict_outputs)
        save_btn.click(
            fn=submit_label,
            inputs=[human_label, last_crop, last_result],
            outputs=[status],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="FER Gradio demo")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help=(
            "Local .pt you bring (weights are not in git). Repeat for an "
            "ensemble. Default: configs/public.yaml paths.best_checkpoint. "
            "FER_CHECKPOINT(S) still win."
        ),
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help=(
            "Load inference.ensemble.members from configs/public.yaml "
            "(does not change enabled in the file). --checkpoint and "
            "FER_CHECKPOINT(S) still win."
        ),
    )
    args = parser.parse_args()
    demo = build_demo(ensemble=args.ensemble, extra_checkpoints=args.checkpoint)
    demo.launch()


if __name__ == "__main__":
    main()
