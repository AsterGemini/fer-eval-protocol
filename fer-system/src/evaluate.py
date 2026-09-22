"""Validation / evaluation: metrics, confusion matrix, CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import create_extra_val_loaders, create_val_loader
from src.inference import ProbabilityEnsemble, load_predictor
from src.model import FERModel
from src.utils import (
    ensure_dir,
    load_config,
    project_root,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("fer.evaluate")


@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    """Run one full pass over ``loader`` and return scalar + per-class metrics.

    Overall ``loss`` uses the training criterion (may be class-weighted).
    Per-class ``loss`` uses unweighted sample-wise CE for interpretability.
    """
    model.eval()
    if isinstance(model, ProbabilityEnsemble):
        model.reset_cascade_stats()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    n_batches = 0

    num_classes = len(class_names)
    class_loss_sum = torch.zeros(num_classes, dtype=torch.float64)
    class_loss_count = torch.zeros(num_classes, dtype=torch.float64)
    # Unweighted CE so per-class loss is comparable across emotions.
    ce_none = nn.CrossEntropyLoss(reduction="none")

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            n_batches += 1

        per_sample = ce_none(logits, labels).detach().cpu()
        labels_cpu = labels.detach().cpu()
        for c in range(num_classes):
            mask = labels_cpu == c
            if mask.any():
                class_loss_sum[c] += float(per_sample[mask].sum())
                class_loss_count[c] += int(mask.sum())

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    class_loss: dict[str, float] = {}
    for i, name in enumerate(class_names):
        if class_loss_count[i] > 0:
            class_loss[name] = float(class_loss_sum[i] / class_loss_count[i])
        else:
            class_loss[name] = float("nan")

    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "loss": class_loss[name],
        }
        for i, name in enumerate(class_names)
    }

    avg_loss = total_loss / max(n_batches, 1) if criterion is not None else None
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    result: dict[str, Any] = {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "class_loss": class_loss,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if isinstance(model, ProbabilityEnsemble):
        result.update(model.cascade_stats())
    return result


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    class_names: list[str],
    save_path: Path,
    normalize: bool = True,
) -> None:
    """Save a confusion-matrix heatmap (counts or row-normalized)."""
    matrix = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, a_min=1.0, a_max=None)
        matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (row-normalized)" if normalize else ""))
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", save_path)


def load_checkpoint_into_model(
    model: FERModel,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load model weights from a training checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = project_root() / path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    logger.info("Loaded checkpoint from %s", path)
    return ckpt if isinstance(ckpt, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FER checkpoint(s) on validation sets"
    )
    parser.add_argument("--config", type=str, default="configs/public.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        required=True,
        help="Checkpoint path; repeat to evaluate a probability ensemble.",
    )
    parser.add_argument(
        "--weight",
        type=float,
        action="append",
        help="Positive member weight; repeat once per checkpoint.",
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help=(
            "Entropy cascade: run later members only on high-entropy rows. "
            "Default is always-on ensemble (METRIC baseline)."
        ),
    )
    parser.add_argument(
        "--max-entropy",
        type=float,
        default=None,
        help="Skip later members when B0 entropy is at most this (nats). Implies --cascade.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override figure/metrics dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger_ = setup_logging("fer.evaluate", log_dir=cfg["paths"]["log_dir"])
    set_seed(int(cfg["seed"]))

    val_loader = create_val_loader(cfg)
    cascade_enabled = True if args.cascade or args.max_entropy is not None else False
    model, device, _ = load_predictor(
        cfg,
        args.checkpoint,
        weights=args.weight,
        cascade_enabled=cascade_enabled,
        max_entropy=args.max_entropy,
    )
    logger_.info("Device: %s", device)

    # Unweighted CE for eval reporting (training may still use a sampler).
    criterion = nn.CrossEntropyLoss()

    metrics = run_evaluation(model, val_loader, criterion, device, cfg["class_names"])
    logger_.info("Val loss=%.4f acc=%.4f macro_f1=%.4f", metrics["loss"], metrics["accuracy"], metrics["macro_f1"])
    if metrics.get("deferral_rate") is not None:
        logger_.info(
            "cascade deferral_rate=%.4f n_deferred=%s/%s enabled=%s",
            metrics["deferral_rate"],
            metrics.get("n_deferred"),
            metrics.get("n_seen"),
            metrics.get("cascade_enabled"),
        )
    class_loss = metrics.get("class_loss") or {}
    if class_loss:
        logger_.info(
            "Val class_loss | %s",
            " ".join(f"{k}={v:.3f}" for k, v in class_loss.items()),
        )
    per_class = metrics.get("per_class") or {}
    if per_class:
        logger_.info(
            "Val class_f1 | %s",
            " ".join(f"{name}={stats['f1']:.3f}" for name, stats in per_class.items()),
        )
    logger_.info("\n%s", metrics["classification_report"])

    out_dir = Path(args.output_dir) if args.output_dir else ensure_dir(cfg["paths"]["figure_dir"])
    if not out_dir.is_absolute():
        out_dir = ensure_dir(out_dir)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        cfg["class_names"],
        out_dir / "confusion_matrix.png",
        normalize=True,
    )

    # Persist JSON metrics + hyperparameters for experiment tracking
    serializable = {
        k: v
        for k, v in metrics.items()
        if k not in {"y_true", "y_pred", "classification_report"}
    }
    report = {
        "hyperparameters": cfg,
        "checkpoint": args.checkpoint[0] if len(args.checkpoint) == 1 else None,
        "checkpoints": args.checkpoint,
        "ensemble_weights": args.weight,
        "cascade_enabled": cascade_enabled,
        "max_entropy": args.max_entropy,
        "metrics": serializable,
        "classification_report": metrics["classification_report"],
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger_.info("Wrote metrics (+ hyperparameters) → %s", metrics_path)

    # Eval-only second views (e.g. RAF-DB val): separate outputs so the
    # primary metrics.json stays byte-comparable for FER2013.
    for name, extra_loader in create_extra_val_loaders(cfg).items():
        em = run_evaluation(model, extra_loader, criterion, device, cfg["class_names"])
        logger_.info(
            "[%s] loss=%.4f acc=%.4f macro_f1=%.4f",
            name,
            em["loss"] if em["loss"] is not None else float("nan"),
            em["accuracy"],
            em["macro_f1"],
        )
        if em.get("deferral_rate") is not None:
            logger_.info(
                "[%s] cascade deferral_rate=%.4f n_deferred=%s/%s",
                name,
                em["deferral_rate"],
                em.get("n_deferred"),
                em.get("n_seen"),
            )
        logger_.info("[%s]\n%s", name, em["classification_report"])
        plot_confusion_matrix(
            em["confusion_matrix"],
            cfg["class_names"],
            out_dir / f"confusion_matrix_{name}.png",
            normalize=True,
        )
        extra_serializable = {
            k: v
            for k, v in em.items()
            if k not in {"y_true", "y_pred", "classification_report"}
        }
        extra_report = {
            "hyperparameters": cfg,
            "checkpoint": args.checkpoint[0] if len(args.checkpoint) == 1 else None,
            "checkpoints": args.checkpoint,
            "ensemble_weights": args.weight,
            "eval_split": name,
            "cascade_enabled": cascade_enabled,
            "max_entropy": args.max_entropy,
            "metrics": extra_serializable,
            "classification_report": em["classification_report"],
        }
        extra_path = out_dir / f"metrics_{name}.json"
        with extra_path.open("w", encoding="utf-8") as f:
            json.dump(extra_report, f, indent=2)
        logger_.info("Wrote %s metrics → %s", name, extra_path)


if __name__ == "__main__":
    main()
