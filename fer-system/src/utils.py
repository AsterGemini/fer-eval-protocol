"""Shared utilities: config loading, reproducibility, and logging."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def project_root() -> Path:
    """Return the fer-system/ root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a nested dict.

    Design choice: keep all hyperparameters in YAML so experiments are
    reproducible and comparable without editing source code (Chip Huyen:
    treat configs as first-class experiment artifacts).
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping, got {type(cfg)}")
    return cfg


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Slight throughput cost; prefer reproducibility for teaching / baselines.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(
    name: str = "fer",
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a logger that writes to stdout and optionally a log file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_path = Path(log_dir)
        if not log_path.is_absolute():
            log_path = project_root() / log_path
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Select the best available device (CUDA > MPS > CPU)."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (relative paths resolved from project root)."""
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    p.mkdir(parents=True, exist_ok=True)
    return p
