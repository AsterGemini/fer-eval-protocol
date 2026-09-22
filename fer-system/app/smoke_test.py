"""METRIC check: on-disk BYO checkpoint → valid top-k JSON (no Kaggle).

Run from fer-system/ after training or pointing FER_CHECKPOINT at a local .pt:

    python app/smoke_test.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = Path(__file__).resolve().parent
# `python app/smoke_test.py` puts app/ on sys.path[0], which shadows the package.
sys.path = [p for p in sys.path if Path(p).resolve() != _APP_DIR]
sys.path.insert(0, str(_ROOT))

from app.app import select_existing_predictor_spec
from src.inference import load_predictor, predict_pil
from src.utils import load_config, setup_logging

REQUIRED_KEYS = ("predicted_label", "confidence", "top_k", "all_probabilities")


def main() -> None:
    cfg = load_config("configs/public.yaml")
    setup_logging("fer.smoke", log_dir=cfg["paths"]["log_dir"])
    checkpoints, weights = select_existing_predictor_spec(cfg)

    t0 = time.perf_counter()
    model, device, transform = load_predictor(cfg, checkpoints, weights=weights)
    load_s = time.perf_counter() - t0
    print(f"load_predictor: {load_s:.2f}s on {device}")
    print(f"checkpoints: {checkpoints}")

    image = Image.new("RGB", (224, 224), color=(120, 90, 70))

    def _run_once() -> tuple[dict, float]:
        t1 = time.perf_counter()
        out = predict_pil(image, model, device, transform, cfg, top_k=3)
        return out, time.perf_counter() - t1

    result, first_s = _run_once()
    _, infer_s = _run_once()
    print(f"predict_pil first: {first_s:.3f}s  steady: {infer_s:.3f}s")
    print(json.dumps(result, indent=2))

    missing = [k for k in REQUIRED_KEYS if k not in result]
    if missing:
        raise AssertionError(f"Result missing keys: {missing}")
    if not result["top_k"]:
        raise AssertionError("top_k is empty")
    if not result["all_probabilities"]:
        raise AssertionError("all_probabilities is empty")

    print(
        f"METRIC ok: generated 224 image → valid top-k "
        f"(first {first_s:.3f}s, steady {infer_s:.3f}s) via {checkpoints}"
    )


if __name__ == "__main__":
    main()
