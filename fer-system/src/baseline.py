"""Trivial baselines: majority class and random chance on the val split.

Establish a hard floor before comparing any learned model (Chip Huyen:
Designing Machine Learning Systems — always measure against a simple baseline).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.dataset import create_val_loader
from src.utils import ensure_dir, load_config, project_root, set_seed, setup_logging

logger = logging.getLogger("fer.baseline")


def _collect_labels(loader: DataLoader) -> list[int]:
    """Gather all labels from a DataLoader without running a model."""
    labels: list[int] = []
    for _, batch_labels in loader:
        labels.extend(batch_labels.tolist())
    return labels


def majority_class_baseline(
    labels: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Always predict the most frequent class in the val split."""
    counts = Counter(labels)
    majority_idx, majority_count = counts.most_common(1)[0]
    majority_name = class_names[majority_idx]
    preds = [majority_idx] * len(labels)
    y_true = np.array(labels)
    y_pred = np.array(preds)
    return {
        "predicted_class": majority_name,
        "predicted_class_idx": int(majority_idx),
        "support": int(majority_count),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_counts": {
            class_names[i]: int(counts.get(i, 0)) for i in range(len(class_names))
        },
    }


def random_chance_baseline(
    labels: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Expected accuracy / macro-F1 under uniform random prediction.

    Accuracy = 1 / C for C classes (uniform).
    Macro-F1: for each class, precision = recall = prior_c under uniform preds
    when labels follow empirical priors; we use a closed-form approximation
    assuming uniform predictions and empirical label priors.
    """
    n_classes = len(class_names)
    n = len(labels)
    counts = Counter(labels)
    priors = np.array([counts.get(i, 0) / max(n, 1) for i in range(n_classes)], dtype=np.float64)

    # Uniform prediction: P(pred=c) = 1/C for all c.
    # Expected accuracy = sum_c P(true=c) * P(pred=c) = 1/C.
    accuracy = 1.0 / n_classes

    # Per-class: TP ≈ n * prior * (1/C), FP ≈ n * (1-prior) * (1/C),
    # FN ≈ n * prior * (1 - 1/C).
    # precision = TP/(TP+FP) = prior; recall = TP/(TP+FN) = 1/C.
    # f1 = 2 * prior * (1/C) / (prior + 1/C).
    f1s: list[float] = []
    for prior in priors:
        if prior <= 0:
            f1s.append(0.0)
            continue
        precision = float(prior)
        recall = 1.0 / n_classes
        f1s.append(2.0 * precision * recall / (precision + recall))

    return {
        "assumption": "uniform_random_predictions",
        "accuracy": float(accuracy),
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": {name: float(f1) for name, f1 in zip(class_names, f1s)},
    }


def run_baselines(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute majority-class and random-chance baselines on the val split."""
    class_names = list(cfg["class_names"])
    val_loader = create_val_loader(cfg)
    labels = _collect_labels(val_loader)
    logger.info("Collected %d val labels across %d classes", len(labels), len(class_names))

    majority = majority_class_baseline(labels, class_names)
    chance = random_chance_baseline(labels, class_names)

    report = {
        "split": "validation",
        "num_samples": len(labels),
        "class_names": class_names,
        "majority_class": majority,
        "random_chance": chance,
        "notes": (
            "Run on primary val split; use as floor for all model comparisons. "
            "Published FER-2013 SOTA is ~0.70–0.73 accuracy (transfer + ensembles)."
        ),
    }
    logger.info(
        "Majority (%s): acc=%.4f macro_f1=%.4f",
        majority["predicted_class"],
        majority["accuracy"],
        majority["macro_f1"],
    )
    logger.info(
        "Random chance: acc=%.4f macro_f1=%.4f",
        chance["accuracy"],
        chance["macro_f1"],
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trivial FER baselines (majority class, random chance)"
    )
    parser.add_argument("--config", type=str, default="configs/public.yaml")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override figure/metrics dir (default: paths.figure_dir)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging("fer.baseline", log_dir=cfg["paths"]["log_dir"])
    set_seed(int(cfg["seed"]))

    report = run_baselines(cfg)

    out_dir = Path(args.output_dir) if args.output_dir else ensure_dir(cfg["paths"]["figure_dir"])
    if not out_dir.is_absolute():
        out_dir = ensure_dir(project_root() / out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "baselines.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote baselines → %s", out_path)


if __name__ == "__main__":
    main()
