"""Single-image FER inference for one model or a probability ensemble.

``load_predictor`` / ``predict_pil`` keep the model in memory for the Gradio
app. ``predict_image`` is the one-shot CLI helper (loads weights per call).
Face detection lives in ``app/app.py`` — this module assumes a face crop.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose

from data.transforms import build_inference_transform
from src.model import FERModel, build_model
from src.utils import get_device, load_config, project_root, set_seed, setup_logging

logger = logging.getLogger("fer.inference")

CheckpointPath = str | Path
CheckpointInput = CheckpointPath | Sequence[CheckpointPath]
_PREPROCESS_KEYS = ("image_size", "mean", "std")


def _predictive_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats. ``probabilities`` is ``[N, C]``."""
    log_p = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    return -(probabilities * log_p).sum(dim=1)


def _log_weighted_mean(
    member_probabilities: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """``member_probabilities`` is ``[M, N, C]``; returns log mean-probs ``[N, C]``."""
    view = weights.to(dtype=member_probabilities.dtype).view(-1, 1, 1)
    mean = (member_probabilities * view).sum(dim=0)
    return mean.clamp_min(torch.finfo(mean.dtype).tiny).log()


def _validate_cascade(enabled: bool, rule: str, max_entropy: float) -> None:
    if not enabled:
        return
    if rule != "entropy":
        raise ValueError(
            f"Unsupported inference.ensemble.cascade.rule={rule!r}; use 'entropy'"
        )
    if not math.isfinite(float(max_entropy)) or float(max_entropy) < 0:
        raise ValueError("cascade max_entropy must be finite and >= 0")


class ProbabilityEnsemble(nn.Module):
    """Combine model probabilities with a normalized weighted mean.

    ``forward`` returns log-probabilities. They remain valid logits for callers
    that apply softmax or cross-entropy, while preserving the exact probability
    average rather than averaging incomparable raw logits.

    When ``cascade_enabled``, member 0 always runs. Later members run only on
    rows whose member-0 predictive entropy exceeds ``max_entropy`` (nats).
    Confident rows keep member-0 probabilities; uncertain rows use the
    weighted mean.
    """

    def __init__(
        self,
        models: Sequence[nn.Module],
        weights: Sequence[float] | None = None,
        member_names: Sequence[str] | None = None,
        cascade_enabled: bool = False,
        max_entropy: float = 0.8,
        cascade_rule: str = "entropy",
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("An ensemble requires at least one model")

        raw_weights = list(weights) if weights is not None else [1.0] * len(models)
        _validate_weights(raw_weights, len(models))
        _validate_cascade(cascade_enabled, cascade_rule, max_entropy)

        names = list(member_names) if member_names is not None else [
            f"model_{index}" for index in range(len(models))
        ]
        if len(names) != len(models):
            raise ValueError(f"Expected {len(models)} member names, got {len(names)}")

        normalized = torch.tensor(raw_weights, dtype=torch.float32)
        normalized /= normalized.sum()
        self.models = nn.ModuleList(models)
        self.register_buffer("weights", normalized)
        self.member_names = names
        self.cascade_enabled = bool(cascade_enabled)
        self.cascade_rule = cascade_rule
        self.max_entropy = float(max_entropy)
        self._n_seen = 0
        self._n_deferred = 0
        self.last_entropy: torch.Tensor | None = None
        self.last_deferred: torch.Tensor | None = None

    def reset_cascade_stats(self) -> None:
        """Clear eval counters (dummy warmup in ``load_predictor`` must not count)."""
        self._n_seen = 0
        self._n_deferred = 0
        self.last_entropy = None
        self.last_deferred = None

    def cascade_stats(self) -> dict[str, Any]:
        seen = int(self._n_seen)
        deferred = int(self._n_deferred)
        rate = (deferred / seen) if seen else None
        return {
            "cascade_enabled": self.cascade_enabled,
            "cascade_rule": self.cascade_rule if self.cascade_enabled else None,
            "max_entropy": self.max_entropy if self.cascade_enabled else None,
            "n_seen": seen,
            "n_deferred": deferred,
            "deferral_rate": rate,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        first = F.softmax(self.models[0](inputs), dim=1)
        batch = int(first.shape[0])
        self._n_seen += batch
        entropy = _predictive_entropy(first)

        if not self.cascade_enabled or len(self.models) == 1:
            rest = [F.softmax(model(inputs), dim=1) for model in self.models[1:]]
            stacked = torch.stack([first, *rest], dim=0) if rest else first.unsqueeze(0)
            self._n_deferred += batch if rest else 0
            self.last_entropy = entropy.detach()
            self.last_deferred = torch.ones(
                batch, dtype=torch.bool, device=inputs.device
            ) if rest else torch.zeros(batch, dtype=torch.bool, device=inputs.device)
            return _log_weighted_mean(stacked, self.weights)

        uncertain = entropy > self.max_entropy
        n_uncertain = int(uncertain.sum().item())
        self._n_deferred += n_uncertain
        self.last_entropy = entropy.detach()
        self.last_deferred = uncertain.detach()

        mean = first.clone()
        if n_uncertain:
            subset = inputs[uncertain]
            rest = [F.softmax(model(subset), dim=1) for model in self.models[1:]]
            stacked = torch.stack([first[uncertain], *rest], dim=0)
            mixed = torch.exp(_log_weighted_mean(stacked, self.weights))
            mean[uncertain] = mixed
        return mean.clamp_min(torch.finfo(mean.dtype).tiny).log()


def _validate_weights(weights: Sequence[float], member_count: int) -> None:
    if len(weights) != member_count:
        raise ValueError(f"Expected {member_count} ensemble weights, got {len(weights)}")
    if any(not math.isfinite(float(weight)) or float(weight) <= 0 for weight in weights):
        raise ValueError("Ensemble weights must be finite and greater than zero")


def _as_checkpoint_list(checkpoint: CheckpointInput) -> list[CheckpointPath]:
    if isinstance(checkpoint, (str, Path)):
        return [checkpoint]
    checkpoints = list(checkpoint)
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    return checkpoints


def _resolve_checkpoint_path(checkpoint: CheckpointPath) -> Path:
    path = Path(checkpoint)
    return path if path.is_absolute() else project_root() / path


def _load_checkpoint(checkpoint: CheckpointPath) -> Any:
    path = _resolve_checkpoint_path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. This repo does not redistribute "
            "trained weights. Train (`python -m src.train --model b0`) or point "
            "--checkpoint / FER_CHECKPOINT at a local .pt you bring."
        )
    # Training checkpoints also contain optimizer state. Keep that unused bulk
    # off the accelerator while extracting model weights.
    return torch.load(path, map_location="cpu", weights_only=False)


def _checkpoint_model_config(
    cfg: dict[str, Any],
    payload: Any,
    checkpoint: CheckpointPath,
    require_contract: bool = False,
) -> dict[str, Any]:
    """Resolve architecture from checkpoint metadata and validate its contract."""
    path = _resolve_checkpoint_path(checkpoint)
    metadata = payload if isinstance(payload, dict) else {}
    saved_cfg = metadata.get("config")
    if not isinstance(saved_cfg, dict):
        if require_contract:
            raise ValueError(
                f"Checkpoint {path} has no saved config; ensemble members require "
                "architecture, class-order, and preprocessing metadata"
            )
        logger.warning(
            "Checkpoint %s has no saved config; assuming runtime architecture, "
            "class order, and preprocessing",
            path,
        )
        return {**cfg["model"], "pretrained": False}

    saved_classes = saved_cfg.get("class_names")
    runtime_classes = list(cfg["class_names"])
    if saved_classes is None and require_contract:
        raise ValueError(f"Checkpoint {path} has no saved class order")
    if saved_classes is not None and list(saved_classes) != runtime_classes:
        raise ValueError(
            f"Checkpoint {path} class order {list(saved_classes)!r} does not match "
            f"runtime class order {runtime_classes!r}"
        )

    saved_data = saved_cfg.get("data") or {}
    runtime_data = cfg["data"]
    missing_preprocess = [key for key in _PREPROCESS_KEYS if key not in saved_data]
    if missing_preprocess and require_contract:
        raise ValueError(
            f"Checkpoint {path} is missing preprocessing metadata: "
            f"{', '.join(missing_preprocess)}"
        )
    mismatches = [
        key
        for key in _PREPROCESS_KEYS
        if key in saved_data and saved_data[key] != runtime_data.get(key)
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={saved_data[key]!r}, runtime={runtime_data.get(key)!r}"
            for key in mismatches
        )
        raise ValueError(f"Checkpoint {path} preprocessing mismatch ({details})")

    saved_model = saved_cfg.get("model")
    if require_contract and (
        not isinstance(saved_model, dict) or not saved_model.get("backbone")
    ):
        raise ValueError(f"Checkpoint {path} has no saved backbone metadata")
    if not isinstance(saved_model, dict):
        saved_model = dict(cfg["model"])
    else:
        saved_model = dict(saved_model)

    saved_backbone = metadata.get("backbone")
    configured_backbone = saved_model.get("backbone")
    if saved_backbone and configured_backbone and saved_backbone != configured_backbone:
        raise ValueError(
            f"Checkpoint {path} has conflicting backbone metadata: "
            f"{saved_backbone!r} != {configured_backbone!r}"
        )
    if saved_backbone:
        saved_model["backbone"] = saved_backbone

    expected_classes = len(runtime_classes)
    saved_num_classes = int(saved_model.get("num_classes", expected_classes))
    if saved_num_classes != expected_classes:
        raise ValueError(
            f"Checkpoint {path} has {saved_num_classes} outputs; "
            f"runtime expects {expected_classes}"
        )
    saved_model["num_classes"] = expected_classes
    saved_model["pretrained"] = False
    return saved_model


def _load_member(
    cfg: dict[str, Any],
    checkpoint: CheckpointPath,
    device: torch.device,
    require_contract: bool = False,
) -> tuple[FERModel, str]:
    path = _resolve_checkpoint_path(checkpoint)
    payload = _load_checkpoint(path)
    model_cfg = _checkpoint_model_config(
        cfg,
        payload,
        path,
        require_contract=require_contract,
    )
    infer_cfg = {**cfg, "model": model_cfg}
    model = build_model(infer_cfg)
    state = (
        payload["model_state_dict"]
        if isinstance(payload, dict) and "model_state_dict" in payload
        else payload
    )
    model.load_state_dict(state)
    del payload, state
    model = model.to(device)
    model.eval()
    logger.info("Loaded %s from %s", model.backbone_name, path)
    return model, f"{model.backbone_name}:{path}"


def cascade_options_from_cfg(
    cfg: dict[str, Any],
    *,
    enabled: bool | None = None,
    max_entropy: float | None = None,
    rule: str | None = None,
) -> tuple[bool, str, float]:
    """Resolve entropy-cascade knobs from config with optional overrides."""
    cascade_cfg = ((cfg.get("inference") or {}).get("ensemble") or {}).get("cascade") or {}
    resolved_enabled = bool(cascade_cfg.get("enabled", False)) if enabled is None else bool(enabled)
    resolved_rule = str(rule if rule is not None else cascade_cfg.get("rule", "entropy"))
    if max_entropy is not None:
        resolved_max = float(max_entropy)
    elif cascade_cfg.get("max_entropy") is not None:
        resolved_max = float(cascade_cfg["max_entropy"])
    else:
        resolved_max = 0.8
    _validate_cascade(resolved_enabled, resolved_rule, resolved_max)
    return resolved_enabled, resolved_rule, resolved_max


def load_predictor(
    cfg: dict[str, Any],
    checkpoint: CheckpointInput,
    weights: Sequence[float] | None = None,
    cascade_enabled: bool | None = None,
    max_entropy: float | None = None,
) -> tuple[nn.Module, torch.device, Compose]:
    """Build member model(s), load weights, and return predictor components.

    Call once at process start; reuse the returned objects for every image.
    Full training checkpoints supply each member's backbone metadata, allowing
    heterogeneous ensembles such as EfficientNet-B0 + EfficientNet-B2.
    """
    device = get_device()
    checkpoints = _as_checkpoint_list(checkpoint)
    if weights is not None:
        _validate_weights(weights, len(checkpoints))
    members_and_names = [
        _load_member(
            cfg,
            path,
            device,
            require_contract=len(checkpoints) > 1,
        )
        for path in checkpoints
    ]
    members = [member for member, _ in members_and_names]
    member_names = [name for _, name in members_and_names]

    model: nn.Module
    if len(members) == 1:
        model = members[0]
    else:
        enabled, cascade_rule, resolved_max = cascade_options_from_cfg(
            cfg,
            enabled=cascade_enabled,
            max_entropy=max_entropy,
        )
        model = ProbabilityEnsemble(
            members,
            weights=weights,
            member_names=member_names,
            cascade_enabled=enabled,
            max_entropy=resolved_max,
            cascade_rule=cascade_rule,
        ).to(device)
        model.eval()

    transform = build_inference_transform(cfg)
    # Prime device kernels so the first user request is not a compile stall.
    image_size = int(cfg["data"]["image_size"])
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        model(dummy)
    if isinstance(model, ProbabilityEnsemble):
        model.reset_cascade_stats()
        if model.cascade_enabled:
            logger.info(
                "Predictor ready on %s with %d member(s); entropy cascade max_entropy=%.3f",
                device,
                len(members),
                model.max_entropy,
            )
        else:
            logger.info("Predictor ready on %s with %d member(s)", device, len(members))
    else:
        logger.info("Predictor ready on %s with %d member(s)", device, len(members))
    return model, device, transform


@torch.no_grad()
def predict_pil(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
    transform: Compose,
    cfg: dict[str, Any],
    top_k: int = 3,
) -> dict[str, Any]:
    """Run inference on a PIL image and return top-k predictions."""
    gray = image.convert("L")
    tensor = transform(gray).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    class_names = cfg["class_names"]
    k = min(top_k, len(class_names))
    top_probs, top_idxs = torch.topk(probs, k=k)

    predictions = [
        {"label": class_names[int(idx)], "probability": float(prob)}
        for prob, idx in zip(top_probs, top_idxs)
    ]
    result = {
        "predicted_label": predictions[0]["label"],
        "confidence": predictions[0]["probability"],
        "top_k": predictions,
        "all_probabilities": {
            name: float(probs[i]) for i, name in enumerate(class_names)
        },
    }
    if isinstance(model, ProbabilityEnsemble):
        result["ensemble"] = {
            "method": "weighted_mean_probabilities",
            "members": list(model.member_names),
            "weights": [float(weight) for weight in model.weights.cpu()],
        }
        if model.cascade_enabled:
            entropy = model.last_entropy
            deferred = model.last_deferred
            result["cascade"] = {
                "rule": model.cascade_rule,
                "max_entropy": model.max_entropy,
                "entropy": None if entropy is None else float(entropy.reshape(-1)[0].cpu()),
                "deferred": None if deferred is None else bool(deferred.reshape(-1)[0].cpu()),
            }
    return result


@torch.no_grad()
def predict_image(
    image_path: str | Path,
    checkpoint: CheckpointInput,
    cfg: dict,
    top_k: int = 3,
    weights: Sequence[float] | None = None,
    cascade_enabled: bool | None = None,
    max_entropy: float | None = None,
) -> dict:
    """Run inference on one image path (loads all members once for this call)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    model, device, transform = load_predictor(
        cfg,
        checkpoint,
        weights=weights,
        cascade_enabled=cascade_enabled,
        max_entropy=max_entropy,
    )
    image = Image.open(path)
    result = predict_pil(image, model, device, transform, cfg, top_k=top_k)
    result["image"] = str(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="FER single-image or ensemble inference")
    parser.add_argument("--config", type=str, default="configs/public.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        required=True,
        help="Checkpoint path; repeat to form a probability ensemble.",
    )
    parser.add_argument(
        "--weight",
        type=float,
        action="append",
        help="Positive member weight; repeat once per checkpoint.",
    )
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="Run later members only when member-0 entropy exceeds --max-entropy.",
    )
    parser.add_argument(
        "--max-entropy",
        type=float,
        default=None,
        help="Skip later members when B0 entropy is at most this (nats). Implies --cascade.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging("fer.inference", log_dir=cfg["paths"]["log_dir"])
    set_seed(int(cfg["seed"]))

    cascade_enabled = True if args.cascade or args.max_entropy is not None else None
    result = predict_image(
        args.image,
        args.checkpoint,
        cfg,
        top_k=args.top_k,
        weights=args.weight,
        cascade_enabled=cascade_enabled,
        max_entropy=args.max_entropy,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        logger.info("Image: %s", result["image"])
        logger.info(
            "Prediction: %s (%.2f%%)",
            result["predicted_label"],
            100 * result["confidence"],
        )
        logger.info("Top-%d:", args.top_k)
        for item in result["top_k"]:
            logger.info("  %-10s %.2f%%", item["label"], 100 * item["probability"])


if __name__ == "__main__":
    main()
