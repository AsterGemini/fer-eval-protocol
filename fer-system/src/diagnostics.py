"""Training diagnostics: loss curves, Karpathy update/weight ratios, run logs."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.utils import ensure_dir

logger = logging.getLogger("fer.diagnostics")

# Karpathy (makemore): healthy update/weight std ratio is ~1e-3.
TARGET_UPDATE_RATIO = 1e-3

# FERModel / EfficientNet-B0 parameter-name groups for per-layer UD.
UD_GROUP_NAMES = ("stem", "last_blocks", "head")
_LAST_BLOCK_PREFIXES = (
    "backbone.blocks.4",
    "backbone.blocks.5",
    "backbone.blocks.6",
    "backbone.conv_head",
    "backbone.bn2",
)


def param_group_name(param_name: str) -> str:
    """Map a ``named_parameters()`` key to stem / last_blocks / head."""
    if param_name.startswith("head."):
        return "head"
    if any(param_name.startswith(p) for p in _LAST_BLOCK_PREFIXES):
        return "last_blocks"
    return "stem"


def snapshot_trainable(model: nn.Module) -> dict[str, torch.Tensor]:
    """CPU clones of trainable params (used before optimizer.step)."""
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def mean_update_to_weight_ratio(
    model: nn.Module,
    before: dict[str, torch.Tensor],
) -> tuple[float | None, dict[str, float | None]]:
    """Mean std(Δw) / std(w) overall and per stem / last_blocks / head.

    Rough guide: ~1e-3 → learning at a sensible scale;
    << 1e-3 → updates too small; >> 1e-3 → possibly unstable.
    """
    all_ratios: list[float] = []
    group_ratios: dict[str, list[float]] = {g: [] for g in UD_GROUP_NAMES}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in before or not param.requires_grad:
                continue
            w = param.detach().float().cpu()
            upd = before[name].float() - w
            if w.numel() <= 1:
                continue
            w_std = float(w.std().clamp_min(1e-8))
            ratio = float(upd.std() / w_std)
            all_ratios.append(ratio)
            group_ratios[param_group_name(name)].append(ratio)

    overall = sum(all_ratios) / len(all_ratios) if all_ratios else None
    by_group: dict[str, float | None] = {
        g: (sum(rs) / len(rs) if rs else None) for g, rs in group_ratios.items()
    }
    return overall, by_group


def plot_loss_curves(history: dict[str, list], save_path: Path) -> None:
    """Plot train/val loss across epochs, plus macro F1 panel.

    When ``history['extra_val']`` holds eval-only second views (e.g. RAF-DB),
    their val loss / macro F1 are overlaid as dashed lines so both validation
    signals can be compared per epoch.
    """
    epochs = history.get("epoch_global", list(range(1, len(history.get("train_loss", [])) + 1)))
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_macro_f1 = history.get("val_macro_f1", [])
    extra_val = history.get("extra_val", [])
    if not train_loss:
        logger.warning("No train_loss in history; skipping loss curve plot.")
        return

    extra_names: list[str] = []
    for row in extra_val:
        if isinstance(row, dict):
            for name in row:
                if name not in extra_names:
                    extra_names.append(name)

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(13, 5))
    ax_loss.plot(epochs, train_loss, label="train", marker="o", markersize=3)
    if val_loss:
        ax_loss.plot(epochs, val_loss, label="val (fer2013)", marker="o", markersize=3)
    for name in extra_names:
        series = [
            (row.get(name) or {}).get("loss") if isinstance(row, dict) else None
            for row in extra_val
        ]
        ax_loss.plot(
            epochs[: len(series)],
            [v if v is not None else float("nan") for v in series],
            label=f"val ({name})",
            marker="o",
            markersize=3,
            linestyle="--",
        )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Train / Val Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    if val_macro_f1:
        ax_f1.plot(epochs, val_macro_f1, label="val (fer2013)", marker="o", markersize=3)
    for name in extra_names:
        series = [
            (row.get(name) or {}).get("macro_f1") if isinstance(row, dict) else None
            for row in extra_val
        ]
        ax_f1.plot(
            epochs[: len(series)],
            [v if v is not None else float("nan") for v in series],
            label=f"val ({name})",
            marker="o",
            markersize=3,
            linestyle="--",
        )
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Macro F1")
    ax_f1.set_title("Validation Macro F1")
    ax_f1.legend()
    ax_f1.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved loss curves → %s", save_path)


def _log10_series(values: list[float | None]) -> list[float]:
    return [
        math.log10(v) if v is not None and v > 0 else float("nan")
        for v in values
    ]


def plot_update_ratio(history: dict[str, list], save_path: Path) -> None:
    """Plot log10 update/weight ratio: overall mean + per-group lines."""
    ratios = history.get("update_ratio", [])
    if not ratios or all(r is None for r in ratios):
        logger.warning("No update_ratio in history; skipping update-ratio plot.")
        return

    epochs = history.get("epoch_global", list(range(1, len(ratios) + 1)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epochs,
        _log10_series(ratios),
        label="mean (all)",
        marker="o",
        markersize=3,
        color="#c45c26",
        linewidth=2,
    )

    by_group = history.get("update_ratio_by_group", [])
    group_colors = {"stem": "#4c72b0", "last_blocks": "#55a868", "head": "#8172b3"}
    if by_group:
        for group in UD_GROUP_NAMES:
            series = [
                row.get(group) if isinstance(row, dict) else None for row in by_group
            ]
            if all(v is None for v in series):
                continue
            ax.plot(
                epochs[: len(series)],
                _log10_series(series),
                label=group,
                marker="o",
                markersize=3,
                color=group_colors.get(group),
            )

    ax.axhline(
        math.log10(TARGET_UPDATE_RATIO),
        color="gray",
        linestyle="--",
        label=f"target log10({TARGET_UPDATE_RATIO:g}) = {math.log10(TARGET_UPDATE_RATIO):g}",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("log10(update / weight std ratio)")
    ax.set_title("Update-to-Weight Ratio (Karpathy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved update-ratio plot → %s", save_path)


def plot_class_loss_curves(
    history: dict[str, list],
    class_names: list[str],
    save_path: Path,
) -> None:
    """Plot per-class validation CE loss over epochs."""
    series = history.get("val_class_loss", [])
    if not series:
        logger.warning("No val_class_loss in history; skipping class-loss plot.")
        return

    epochs = history.get("epoch_global", list(range(1, len(series) + 1)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in class_names:
        ys = [float(row.get(name, float("nan"))) if isinstance(row, dict) else float("nan") for row in series]
        ax.plot(epochs, ys, label=name, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val CE loss (unweighted)")
    ax.set_title("Per-Class Validation Loss")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved class-loss curves → %s", save_path)


def empty_history() -> dict[str, list]:
    return {
        "epoch_global": [],
        "phase": [],
        "epoch_in_phase": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_class_loss": [],
        "val_class_f1": [],
        "lr": [],
        "update_ratio": [],
        "update_ratio_by_group": [],
        "extra_val": [],
    }


def append_epoch(
    history: dict[str, list],
    *,
    phase: str,
    epoch_in_phase: int,
    train_loss: float,
    val_loss: float | None,
    val_accuracy: float,
    val_macro_f1: float,
    lr: float,
    update_ratio: float | None,
    val_class_loss: dict[str, float] | None = None,
    val_class_f1: dict[str, float] | None = None,
    update_ratio_by_group: dict[str, float | None] | None = None,
    extra_val: dict[str, dict[str, Any]] | None = None,
) -> None:
    history["epoch_global"].append(len(history["epoch_global"]) + 1)
    history["phase"].append(phase)
    history["epoch_in_phase"].append(epoch_in_phase)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)
    history["val_macro_f1"].append(val_macro_f1)
    history["val_class_loss"].append(dict(val_class_loss or {}))
    history["val_class_f1"].append(dict(val_class_f1 or {}))
    history["lr"].append(lr)
    history["update_ratio"].append(update_ratio)
    history["update_ratio_by_group"].append(
        {g: (update_ratio_by_group or {}).get(g) for g in UD_GROUP_NAMES}
    )
    history.setdefault("extra_val", []).append(dict(extra_val or {}))


def save_run_report(
    *,
    cfg: dict[str, Any],
    history: dict[str, list],
    best_metric: float,
    best_checkpoint: str,
    figure_dir: str | Path,
    log_dir: str | Path,
    extra_metrics: dict[str, Any] | None = None,
    run_stamp: str | None = None,
) -> Path:
    """Write experiment JSON: hyperparameters + history + summary metrics.

    ``run_stamp`` should match the per-run checkpoint stamp when provided so
    ``run_<stamp>.json`` and ``run_<stamp>.pt`` stay paired.
    """
    fig_dir = ensure_dir(figure_dir)
    logs = ensure_dir(log_dir)
    stamp = run_stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    report = {
        "timestamp_utc": stamp,
        "hyperparameters": cfg,
        "best_metric_name": cfg.get("training", {}).get("monitor_metric", "macro_f1"),
        "best_metric_value": best_metric,
        "best_checkpoint": best_checkpoint,
        "history": history,
        "metrics": extra_metrics or {},
    }

    run_path = logs / f"run_{stamp}.json"
    with run_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Also write a stable latest pointer for evaluate / quick inspection
    latest_path = fig_dir / "metrics.json"
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Wrote run report → %s", run_path)
    logger.info("Wrote metrics+hyperparameters → %s", latest_path)
    return run_path
