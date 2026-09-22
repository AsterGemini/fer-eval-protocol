"""FER training loop: checkpoint by monitor metric, optional Pareto class-F1 veto."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from tqdm import tqdm

from src.dataset import create_dataloaders, create_extra_val_loaders
from src.diagnostics import (
    UD_GROUP_NAMES,
    append_epoch,
    empty_history,
    mean_update_to_weight_ratio,
    plot_class_loss_curves,
    plot_loss_curves,
    plot_update_ratio,
    save_run_report,
    snapshot_trainable,
)
from src.evaluate import load_checkpoint_into_model, run_evaluation
from src.model import FERModel, build_model
from src.pareto_gate import evaluate_pareto, load_baseline
from src.utils import (
    ensure_dir,
    get_device,
    load_config,
    project_root,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("fer.train")

MODEL_ALIASES = {
    "b0": "efficientnet_b0",
    "b1": "efficientnet_b1",
    "b2": "efficientnet_b2",
}


def model_artifacts_root(backbone: str) -> Path:
    """Checkpoints, logs, and figures for one backbone family."""
    return Path("artifacts") / "models" / str(backbone)


_LEGACY_B0_BEST = Path("artifacts") / "checkpoints" / "best.pt"


def warn_if_legacy_b0_left_behind(cfg: dict[str, Any]) -> None:
    """Warn if a pre-unification B0 checkpoint is still at the old root."""
    best = Path(cfg["paths"]["best_checkpoint"])
    if best.exists() or not _LEGACY_B0_BEST.exists():
        return
    logger.warning(
        "Leftover B0 checkpoint at %s is no longer the resume path. "
        "Move it to %s (and move logs/ and figures/ into "
        "artifacts/models/efficientnet_b0/) or this run will not resume those weights.",
        _LEGACY_B0_BEST,
        best,
    )


def resolve_model_selection(
    cfg: dict[str, Any],
    model_alias: str | None = None,
    artifacts_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a config resolved for one CLI-selected model.

    Every CLI alias writes under ``artifacts/models/<backbone>/`` so families
    cannot load or overwrite each other. Custom configs with no ``--model``
    keep their yaml paths (A/B isolation). B0 keeps yaml ``pareto_gate`` if
    set; other aliases disable the in-run veto so a candidate checkpoint is
    available for post-training comparison.
    """
    resolved = copy.deepcopy(cfg)

    if model_alias is not None:
        try:
            backbone = MODEL_ALIASES[model_alias]
        except KeyError as exc:
            choices = ", ".join(sorted(MODEL_ALIASES))
            raise ValueError(f"Unknown model '{model_alias}'. Choose one of: {choices}") from exc
        resolved["model"]["backbone"] = backbone
    else:
        backbone = str(resolved["model"]["backbone"])

    isolated_new_family = model_alias is not None and model_alias != "b0"
    root: Path | None = Path(artifacts_root) if artifacts_root else None
    if root is None and model_alias is not None:
        root = model_artifacts_root(backbone)

    if root is not None:
        resolved["paths"].update(
            {
                "checkpoint_dir": str(root / "checkpoints"),
                "log_dir": str(root / "logs"),
                "figure_dir": str(root / "figures"),
                "best_checkpoint": str(root / "checkpoints" / "best.pt"),
            }
        )

    if isolated_new_family:
        # A new architecture needs its own best candidate even when it does not
        # beat every B0 Pareto class floor. Compare it with B0 after training.
        resolved["training"]["pareto_gate"] = False

    return resolved


def model_selection_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    """Small serializable summary for the CLI dry-run."""
    return {
        "backbone": cfg["model"]["backbone"],
        "pretrained": bool(cfg["model"].get("pretrained", True)),
        "image_size": int(cfg["data"]["image_size"]),
        "epochs": int(cfg["training"]["epochs"]),
        "early_stopping_patience": int(
            cfg["training"].get("early_stopping_patience", 0)
        ),
        "resume_from_best": bool(cfg["training"].get("resume_from_best", False)),
        "pareto_gate": bool(cfg["training"].get("pareto_gate", False)),
        "unfreeze_last_n_blocks": cfg["training"].get("unfreeze_last_n_blocks"),
        "freeze_phase": freeze_profile_name(cfg["training"]),
        "paths": dict(cfg["paths"]),
    }


def freeze_profile_name(training_cfg: dict[str, Any]) -> str:
    """Name of the freeze profile implied by config (no model build)."""
    if bool(training_cfg.get("freeze_backbone", False)):
        return "head_only"
    last_n = training_cfg.get("unfreeze_last_n_blocks")
    if last_n in (None, ""):
        return "full"
    return f"partial_last_{int(last_n)}"


def configure_backbone_trainability(
    model: FERModel, training_cfg: dict[str, Any]
) -> str:
    """Apply the configured freeze profile and return its run-log phase name."""
    freeze_backbone = bool(training_cfg.get("freeze_backbone", False))
    last_n = training_cfg.get("unfreeze_last_n_blocks")

    if freeze_backbone:
        if last_n not in (None, ""):
            raise ValueError(
                "freeze_backbone=true conflicts with unfreeze_last_n_blocks"
            )
        model.freeze_backbone()
        return freeze_profile_name(training_cfg)

    if last_n in (None, ""):
        model.unfreeze_backbone(last_n_blocks=None)
        return freeze_profile_name(training_cfg)

    if isinstance(last_n, bool):
        raise ValueError("unfreeze_last_n_blocks must be a positive integer or null")
    try:
        last_n_int = int(last_n)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "unfreeze_last_n_blocks must be a positive integer or null"
        ) from exc
    if last_n_int <= 0:
        raise ValueError("unfreeze_last_n_blocks must be a positive integer or null")

    model.unfreeze_backbone(last_n_blocks=last_n_int)
    return freeze_profile_name(training_cfg)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, cfg: dict[str, Any]) -> AdamW:
    """AdamW over trainable parameters only."""
    params = [p for p in model.parameters() if p.requires_grad]
    betas = tuple(cfg.get("optimizer", {}).get("betas", [0.9, 0.999]))
    return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def build_scheduler(
    optimizer: AdamW,
    epochs: int,
    cfg: dict[str, Any],
) -> LRScheduler:
    """Build a cosine LR schedule (T_max = epochs).

    Optional knobs under ``cfg['scheduler']``:
    - ``eta_min``: absolute floor (default 0)
    - ``eta_min_ratio``: floor as a fraction of the base LR (e.g. 0.1 →
      decay only down to 10% of starting LR). Takes precedence over a lower
      absolute ``eta_min`` when both are set.
    """
    sched_cfg = cfg.get("scheduler", {}) or {}
    name = str(sched_cfg.get("name", "cosine")).lower()
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler.name={name!r}; only 'cosine' is implemented")

    t_max = max(int(epochs), 1)
    eta_min = float(sched_cfg.get("eta_min", 0.0))
    eta_min_ratio = sched_cfg.get("eta_min_ratio")
    if eta_min_ratio is not None:
        base_lr = float(optimizer.param_groups[0]["lr"])
        eta_min = max(eta_min, base_lr * float(eta_min_ratio))

    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    logger.info("Scheduler: cosine | T_max=%d | eta_min=%g", t_max, eta_min)
    return scheduler


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a checkpoint path relative to the project root when needed."""
    ckpt_path = Path(path)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root() / ckpt_path
    return ckpt_path


def per_run_checkpoint_path(checkpoint_dir: str | Path, run_stamp: str) -> Path:
    """Timestamped sibling of best.pt for this training run."""
    ckpt_dir = resolve_checkpoint_path(checkpoint_dir)
    return ckpt_dir / f"run_{run_stamp}.pt"


def try_resume_from_best(
    model: FERModel,
    best_path: str | Path,
    device: torch.device,
    monitor: str,
    enabled: bool,
) -> tuple[float, str | None, float | None]:
    """Load model weights from best.pt when present and resume is enabled.

    Returns ``(best_metric, resumed_from, prior_best_metric)``.
    Optimizer / scheduler are intentionally not restored (fresh each run).
    """
    if not enabled:
        logger.info("resume_from_best=false; starting from pretrained backbone.")
        return -1.0, None, None

    path = resolve_checkpoint_path(best_path)
    if not path.is_file():
        logger.info(
            "No checkpoint at %s; starting from pretrained backbone.",
            path,
        )
        return -1.0, None, None

    try:
        ckpt = load_checkpoint_into_model(model, path, device)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load model weights from {path}. "
            "Check that backbone / num_classes match the checkpoint architecture."
        ) from exc

    metrics = ckpt.get("metrics") or {}
    prior = resolve_monitor_value(monitor, metrics, metrics.get("extra_val"))
    best_metric = prior if prior is not None else -1.0
    resumed_from = str(path)
    logger.info(
        "Resumed model weights from %s | prior %s=%.4f | ckpt epoch=%s phase=%s",
        path,
        monitor,
        best_metric,
        ckpt.get("epoch"),
        ckpt.get("phase"),
    )
    return best_metric, resumed_from, best_metric if prior is not None else None


def _amp_device_type(device: torch.device) -> str | None:
    """Return autocast device type when AMP is supported, else None."""
    if device.type in {"cuda", "cpu", "mps"}:
        return device.type
    return None


def resolve_monitor_value(
    monitor: str,
    metrics: dict[str, Any],
    extra_val_metrics: dict[str, dict[str, Any]] | None = None,
) -> float | None:
    """Resolve ``monitor_metric`` against primary or extra val metrics.

    A bare key (``macro_f1``) reads the primary val split; ``<source>.<key>``
    (e.g. ``raf_db.macro_f1``) reads an eval-only extra val set.
    """
    if "." not in monitor:
        value = metrics.get(monitor)
        return float(value) if value is not None else None
    source, key = monitor.split(".", 1)
    value = (extra_val_metrics or {}).get(source, {}).get(key)
    return float(value) if value is not None else None


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None,
    update_ratio_every: int = 10,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[float, float | None, dict[str, float | None] | None]:
    """Single training epoch; returns (loss, mean UD ratio, per-group UD ratios)."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    ratio_samples: list[float] = []
    group_samples: dict[str, list[float]] = {g: [] for g in UD_GROUP_NAMES}
    amp_type = _amp_device_type(device) if use_amp else None

    for step, (images, labels) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp_type is not None:
            with torch.autocast(device_type=amp_type):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        else:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        # Karpathy diagnostic: sample std(Δw)/std(w) every N steps (cheap-ish).
        take_ratio = update_ratio_every > 0 and (step % update_ratio_every == 0)
        before = snapshot_trainable(model) if take_ratio else None

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if before is not None:
            ratio, by_group = mean_update_to_weight_ratio(model, before)
            if ratio is not None:
                ratio_samples.append(ratio)
            for g, g_ratio in by_group.items():
                if g_ratio is not None:
                    group_samples[g].append(g_ratio)

        running_loss += float(loss.item())
        n_batches += 1

    mean_loss = running_loss / max(n_batches, 1)
    mean_ratio = sum(ratio_samples) / len(ratio_samples) if ratio_samples else None
    mean_by_group: dict[str, float | None] | None = None
    if any(group_samples[g] for g in UD_GROUP_NAMES):
        mean_by_group = {
            g: (sum(rs) / len(rs) if rs else None) for g, rs in group_samples.items()
        }
    return mean_loss, mean_ratio, mean_by_group


def save_checkpoint(
    path: str | Path,
    model: FERModel,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    epoch: int,
    phase: str,
    metrics: dict[str, Any],
    cfg: dict[str, Any],
    run_stamp: str | None = None,
    extra_val_metrics: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Persist a full training checkpoint for resume / evaluation."""
    ckpt_path = resolve_checkpoint_path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "phase": phase,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": {
            "loss": metrics.get("loss"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "per_class": metrics.get("per_class"),
            "class_loss": metrics.get("class_loss"),
            # Eval-only second views (e.g. raf_db) so resume can read prior best
            # when monitor_metric points at an extra val set.
            "extra_val": dict(extra_val_metrics or {}),
        },
        "config": cfg,
        "backbone": model.backbone_name,
        "run_stamp": run_stamp,
    }
    torch.save(payload, ckpt_path)
    logger.info("Saved checkpoint → %s (macro_f1=%.4f)", ckpt_path, metrics.get("macro_f1", 0.0))


def run_training(
    phase_name: str,
    model: FERModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: dict[str, Any],
    best_metric: float,
    best_path: Any,
    history: dict[str, list],
    run_stamp: str,
    eval_criterion: nn.Module | None = None,
    extra_val_loaders: dict[str, torch.utils.data.DataLoader] | None = None,
) -> tuple[float, bool]:
    """Run the training loop over all trainable parameters.

    Returns ``(best_metric, saved_per_run_checkpoint)`` where the bool is True
    if this run wrote ``run_{stamp}.pt`` after an improvement.
    """
    train_cfg = cfg["training"]
    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    patience = int(train_cfg.get("early_stopping_patience", 5))
    grad_clip = train_cfg.get("grad_clip_norm")
    monitor = train_cfg.get("monitor_metric", "macro_f1")
    pareto_enabled = bool(train_cfg.get("pareto_gate", False))
    pareto_baseline = load_baseline(train_cfg.get("pareto_baseline")) if pareto_enabled else None
    update_ratio_every = int(train_cfg.get("update_ratio_every", 10))
    use_amp = bool(train_cfg.get("use_amp", True))
    per_run_path = per_run_checkpoint_path(cfg["paths"]["checkpoint_dir"], run_stamp)
    saved_per_run = False

    optimizer = build_optimizer(model, lr, weight_decay, cfg)
    scheduler = build_scheduler(optimizer, epochs, cfg)
    # GradScaler is CUDA-only; MPS/CPU use autocast without scaling when AMP is on.
    scaler: torch.cuda.amp.GradScaler | None = None
    if use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    trainable, total = model.trainable_parameter_count()
    logger.info(
        "=== %s | epochs=%d lr=%g amp=%s | trainable=%d / %d params ===",
        phase_name,
        epochs,
        lr,
        use_amp,
        trainable,
        total,
    )

    if "." in monitor:
        monitor_source = monitor.split(".", 1)[0]
        if not extra_val_loaders or monitor_source not in extra_val_loaders:
            raise ValueError(
                f"monitor_metric={monitor!r} requires an extra val loader named "
                f"{monitor_source!r}, but available extra val loaders are "
                f"{sorted((extra_val_loaders or {}).keys())}. Set "
                f"data.extra_train_sources[].val_dir for that source."
            )

    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, update_ratio, update_ratio_by_group = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip,
            update_ratio_every=update_ratio_every,
            use_amp=use_amp,
            scaler=scaler,
        )
        metrics = run_evaluation(
            model, val_loader, eval_criterion or criterion, device, cfg["class_names"]
        )

        # Eval-only second views (e.g. RAF-DB val). Dotted monitor_metric
        # (raf_db.macro_f1) may checkpoint on these; class_f1 is for diagnostics.
        extra_val_metrics: dict[str, dict[str, Any]] = {}
        for name, extra_loader in (extra_val_loaders or {}).items():
            em = run_evaluation(
                model, extra_loader, eval_criterion or criterion, device, cfg["class_names"]
            )
            extra_per_class = em.get("per_class") or {}
            extra_class_f1 = {
                cls: float(stats["f1"]) for cls, stats in extra_per_class.items()
            }
            extra_val_metrics[name] = {
                "loss": em["loss"],
                "accuracy": em["accuracy"],
                "macro_f1": em["macro_f1"],
                "class_loss": em.get("class_loss") or {},
                "class_f1": extra_class_f1,
            }
            logger.info(
                "%s val loss=%.4f acc=%.4f macro_f1=%.4f",
                name,
                em["loss"] if em["loss"] is not None else float("nan"),
                em["accuracy"],
                em["macro_f1"],
            )
            if extra_class_f1:
                extra_f1_str = " ".join(
                    f"{cls}={f1:.3f}" for cls, f1 in extra_class_f1.items()
                )
                logger.info("%s val class_f1 | %s", name, extra_f1_str)
        scheduler.step()

        elapsed = time.time() - t0
        ratio_str = f"{update_ratio:.2e}" if update_ratio is not None else "n/a"
        logger.info(
            "%s epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | "
            "acc=%.4f | macro_f1=%.4f | upd/w=%s | lr=%.2e | %.1fs",
            phase_name,
            epoch,
            epochs,
            train_loss,
            metrics["loss"] if metrics["loss"] is not None else float("nan"),
            metrics["accuracy"],
            metrics["macro_f1"],
            ratio_str,
            optimizer.param_groups[0]["lr"],
            elapsed,
        )
        if update_ratio_by_group:
            group_str = " ".join(
                f"{g}="
                + (f"{update_ratio_by_group[g]:.2e}" if update_ratio_by_group[g] is not None else "n/a")
                for g in UD_GROUP_NAMES
            )
            logger.info("upd/w by group | %s", group_str)

        class_loss = metrics.get("class_loss") or {}
        if class_loss:
            class_loss_str = " ".join(f"{k}={v:.3f}" for k, v in class_loss.items())
            logger.info("val class_loss | %s", class_loss_str)

        per_class = metrics.get("per_class") or {}
        class_f1 = {
            name: float(stats["f1"]) for name, stats in per_class.items()
        }
        if class_f1:
            class_f1_str = " ".join(f"{name}={f1:.3f}" for name, f1 in class_f1.items())
            logger.info("val class_f1 | %s", class_f1_str)

        append_epoch(
            history,
            phase=phase_name,
            epoch_in_phase=epoch,
            train_loss=train_loss,
            val_loss=metrics["loss"],
            val_accuracy=metrics["accuracy"],
            val_macro_f1=metrics["macro_f1"],
            lr=float(optimizer.param_groups[0]["lr"]),
            update_ratio=update_ratio,
            val_class_loss=class_loss,
            val_class_f1=class_f1,
            update_ratio_by_group=update_ratio_by_group,
            extra_val=extra_val_metrics,
        )

        current = resolve_monitor_value(monitor, metrics, extra_val_metrics)
        if current is None:
            raise KeyError(
                f"monitor_metric={monitor!r} not found in val metrics; "
                f"extra val sources this epoch: {sorted(extra_val_metrics)}"
            )

        pareto_ok = True
        if pareto_enabled:
            if pareto_baseline is None:
                raise RuntimeError("pareto_gate=true but baseline failed to load")
            split = str(pareto_baseline.get("split") or "raf_db")
            candidate_f1 = (extra_val_metrics.get(split) or {}).get("class_f1") or {}
            pareto = evaluate_pareto(
                {k: float(v) for k, v in pareto_baseline["class_f1"].items()},
                candidate_f1,
                weak_threshold=float(pareto_baseline.get("weak_threshold", 0.8)),
                fer_acc=metrics.get("accuracy"),
                fer_acc_floor=pareto_baseline.get("fer_val_accuracy_floor"),
            )
            pareto_ok = pareto.ok
            if not pareto.ok:
                logger.info("Pareto gate veto | %s", "; ".join(pareto.failures))

        if current > best_metric and pareto_ok:
            best_metric = current
            epochs_without_improve = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                phase_name,
                metrics,
                cfg,
                run_stamp=run_stamp,
                extra_val_metrics=extra_val_metrics,
            )
            save_checkpoint(
                per_run_path,
                model,
                optimizer,
                scheduler,
                epoch,
                phase_name,
                metrics,
                cfg,
                run_stamp=run_stamp,
                extra_val_metrics=extra_val_metrics,
            )
            saved_per_run = True
        else:
            epochs_without_improve += 1
            if pareto_enabled and current > best_metric and not pareto_ok:
                logger.info(
                    "Monitor rose but Pareto veto; no checkpoint for %d epoch(s) (best=%.4f).",
                    epochs_without_improve,
                    best_metric,
                )
            else:
                logger.info(
                    "No improvement in %s for %d epoch(s) (best=%.4f).",
                    monitor,
                    epochs_without_improve,
                    best_metric,
                )

        if epochs_without_improve >= patience:
            logger.info("Early stopping triggered (patience=%d).", patience)
            break

    return best_metric, saved_per_run


def train(cfg: dict[str, Any]) -> None:
    """Train FER. freeze_backbone=true keeps the resumed backbone fixed."""
    set_seed(int(cfg["seed"]))
    device = get_device()
    logger.info("Device: %s", device)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger.info("Run stamp: %s", run_stamp)

    ensure_dir(cfg["paths"]["checkpoint_dir"])
    ensure_dir(cfg["paths"]["log_dir"])
    fig_dir = ensure_dir(cfg["paths"]["figure_dir"])

    train_loader, val_loader, class_weights, weight_info = create_dataloaders(cfg)

    model = build_model(cfg).to(device)

    extra_val_loaders = create_extra_val_loaders(cfg)

    best_path = cfg["paths"]["best_checkpoint"]
    monitor = cfg["training"].get("monitor_metric", "macro_f1")
    history = empty_history()

    best_metric, resumed_from, prior_best_metric = try_resume_from_best(
        model,
        best_path,
        device,
        monitor=monitor,
        enabled=bool(cfg["training"].get("resume_from_best", False)),
    )

    phase_name = configure_backbone_trainability(model, cfg["training"])

    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    if class_weights is not None:
        criterion: nn.Module = nn.CrossEntropyLoss(
            weight=class_weights.to(device), label_smoothing=label_smoothing
        )
        logger.info(
            "Using class-weighted CrossEntropyLoss (label_smoothing=%.3f).",
            label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        logger.info("Using CrossEntropyLoss (label_smoothing=%.3f).", label_smoothing)

    # Eval uses unweighted, unsmoothed CE so val loss stays comparable across runs
    # (the training criterion is class-weighted and possibly smoothed).
    eval_criterion = nn.CrossEntropyLoss()

    best_metric, saved_per_run = run_training(
        phase_name=phase_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        cfg=cfg,
        best_metric=best_metric,
        best_path=best_path,
        history=history,
        run_stamp=run_stamp,
        eval_criterion=eval_criterion,
        extra_val_loaders=extra_val_loaders,
    )

    per_run_rel = str(Path(cfg["paths"]["checkpoint_dir"]) / f"run_{run_stamp}.pt")
    # Diagnostics plots + experiment report (hyperparameters + history)
    plot_loss_curves(history, fig_dir / "loss_curves.png")
    plot_update_ratio(history, fig_dir / "update_ratio.png")
    plot_class_loss_curves(history, cfg["class_names"], fig_dir / "class_loss_curves.png")
    save_run_report(
        cfg=cfg,
        history=history,
        best_metric=best_metric,
        best_checkpoint=str(best_path),
        figure_dir=fig_dir,
        log_dir=cfg["paths"]["log_dir"],
        run_stamp=run_stamp,
        extra_metrics={
            "monitor_metric": monitor,
            "best_monitor_value": best_metric,
            "best_macro_f1": best_metric if monitor == "macro_f1" else None,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else None,
            "final_val_macro_f1": history["val_macro_f1"][-1] if history["val_macro_f1"] else None,
            "final_val_class_loss": history["val_class_loss"][-1] if history["val_class_loss"] else None,
            "final_val_class_f1": history["val_class_f1"][-1] if history["val_class_f1"] else None,
            "final_extra_val": history["extra_val"][-1] if history.get("extra_val") else None,
            "resumed_from": resumed_from,
            "prior_best_metric": prior_best_metric,
            "per_run_checkpoint": per_run_rel if saved_per_run else None,
            "run_stamp": run_stamp,
            # Resolved loss weights (auto + overrides) for experiment comparison
            **weight_info,
        },
    )

    logger.info("Training complete. Best %s=%.4f → %s", monitor, best_metric, best_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a selectable FER transfer-learning model"
    )
    parser.add_argument("--config", type=str, default="configs/public.yaml")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_ALIASES),
        help=(
            "Model alias. Each alias stores checkpoints under "
            "artifacts/models/<backbone>/; b1/b2 start from ImageNet weights. "
            "b0 keeps the held Pareto gate."
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        help="Optional isolated root containing checkpoints/, logs/, and figures/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved model, resume policy, and paths without training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override training.epochs. Leave unset to keep configs/public.yaml. "
            "Vast B2: 200 with --patience 16 so cosine stretches and "
            "early-stop can still halt a plateau."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=(
            "Override training.early_stopping_patience. Colab default is 4; "
            "a 200-epoch Vast job needs a higher cap or it will stop after "
            "four epochs without a monitor beat."
        ),
    )
    parser.add_argument(
        "--full-backbone",
        action="store_true",
        help=(
            "Unfreeze the entire backbone (unfreeze_last_n_blocks=null). "
            "ImageNet-start B2: last-4 leaves EfficientNet blocks frozen. "
            "Do not also raise lr."
        ),
    )
    args = parser.parse_args()

    cfg = resolve_model_selection(
        load_config(args.config),
        model_alias=args.model,
        artifacts_root=args.artifacts_root,
    )
    warn_if_legacy_b0_left_behind(cfg)
    if args.epochs is not None:
        if args.epochs < 1:
            raise SystemExit("--epochs must be >= 1")
        cfg["training"]["epochs"] = int(args.epochs)
    if args.patience is not None:
        if args.patience < 1:
            raise SystemExit("--patience must be >= 1")
        cfg["training"]["early_stopping_patience"] = int(args.patience)
    if args.full_backbone:
        if bool(cfg["training"].get("freeze_backbone", False)):
            raise SystemExit("--full-backbone conflicts with freeze_backbone=true")
        cfg["training"]["unfreeze_last_n_blocks"] = None
    if args.dry_run:
        print(json.dumps(model_selection_summary(cfg), indent=2))
        return

    setup_logging("fer.train", log_dir=cfg["paths"]["log_dir"])
    setup_logging("fer.evaluate", log_dir=cfg["paths"]["log_dir"])
    setup_logging("fer.dataset", log_dir=cfg["paths"]["log_dir"])
    setup_logging("fer.model", log_dir=cfg["paths"]["log_dir"])
    setup_logging("fer.download", log_dir=cfg["paths"]["log_dir"])
    setup_logging("fer.diagnostics", log_dir=cfg["paths"]["log_dir"])

    train(cfg)


if __name__ == "__main__":
    main()
