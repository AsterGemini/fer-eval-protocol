"""FER dataset, class weights, and imbalance-aware samplers."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from data.download import (
    describe_training_sources,
    get_dataset_paths,
    get_extra_train_sources,
    load_manifest,
)
from data.transforms import build_transforms
from src.val_corrections import apply_corrections_from_cfg

logger = logging.getLogger("fer.dataset")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FERDataset(Dataset):
    """Folder-based FER dataset: ``root/<class_name>/*.jpg``.

    Expects an ImageFolder-compatible tree. Class indices follow
    ``cfg['class_names']`` order so train/val/eval stay aligned.

    ``folder_map`` remaps canonical class names → on-disk folder names when a
    source uses different spelling (e.g. ``surprise`` → ``surprised``).
    """

    def __init__(
        self,
        root: str | Path,
        class_names: list[str],
        transform: Callable | None = None,
        folder_map: dict[str, str] | None = None,
        source_name: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.transform = transform
        self.folder_map = dict(folder_map) if folder_map else {}
        self.source_name = source_name or str(self.root)

        self.samples: list[tuple[Path, int]] = []
        self._scan()

    def _scan(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        for name in self.class_names:
            folder = self.folder_map.get(name, name)
            class_dir = self.root / folder
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[name]))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

        counts = Counter(label for _, label in self.samples)
        logger.info(
            "Loaded %d samples from %s (%s) | per-class: %s",
            len(self.samples),
            self.source_name,
            self.root,
            {self.class_names[i]: counts[i] for i in range(len(self.class_names))},
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        # Convert to RGB after open so grayscale sources become 3-channel PIL images
        # before transforms (Grayscale(num_output_channels=3) is also applied).
        image = Image.open(path).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def targets(self) -> list[int]:
        return [label for _, label in self.samples]


def compute_class_weights(
    targets: list[int],
    num_classes: int,
    class_names: list[str] | None = None,
    overrides: dict[str, float] | None = None,
    multipliers: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse-frequency class weights for CrossEntropyLoss.

    Rare classes (e.g. disgust) get higher weight so the loss does not ignore them.
    Weights are normalized to mean 1.0, then optional per-class ``overrides``
    (absolute replace) and ``multipliers`` (scale in place) from config are
    applied. Multipliers run after overrides and are not re-normalized.

    Returns ``(final_weights, auto_weights)`` where ``auto_weights`` are
    pre-override / pre-multiplier.
    """
    counts = torch.bincount(torch.tensor(targets, dtype=torch.long), minlength=num_classes).float()
    # Avoid division by zero if a class is somehow missing from the split.
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    auto_weights = weights.clone()

    if overrides or multipliers:
        if not class_names:
            raise ValueError(
                "class_names required when applying class_weight_overrides "
                "or class_weight_multipliers"
            )
        name_to_idx = {name: i for i, name in enumerate(class_names)}
        if overrides:
            for name, value in overrides.items():
                if name not in name_to_idx:
                    raise KeyError(
                        f"Unknown class in class_weight_overrides: {name!r}. "
                        f"Expected one of {list(class_names)}"
                    )
                weights[name_to_idx[name]] = float(value)
        if multipliers:
            for name, value in multipliers.items():
                if name not in name_to_idx:
                    raise KeyError(
                        f"Unknown class in class_weight_multipliers: {name!r}. "
                        f"Expected one of {list(class_names)}"
                    )
                weights[name_to_idx[name]] *= float(value)

    return weights, auto_weights


def class_weights_to_dict(
    weights: torch.Tensor | None,
    class_names: list[str],
) -> dict[str, float] | None:
    """Convert a weight tensor to a name→value map for logging / metrics."""
    if weights is None:
        return None
    return {name: float(w) for name, w in zip(class_names, weights)}


def build_weighted_sampler(targets: list[int]) -> WeightedRandomSampler:
    """Sample inversely to class frequency so each batch is more balanced."""
    counts = Counter(targets)
    sample_weights = [1.0 / counts[t] for t in targets]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def _train_targets(dataset: Dataset) -> list[int]:
    """Collect labels from a FERDataset or ConcatDataset of FERDatasets."""
    if isinstance(dataset, FERDataset):
        return list(dataset.targets)
    if isinstance(dataset, ConcatDataset):
        targets: list[int] = []
        for sub in dataset.datasets:
            targets.extend(_train_targets(sub))
        return targets
    raise TypeError(f"Unsupported dataset type for targets: {type(dataset)!r}")


def create_dataloaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader, torch.Tensor | None, dict[str, Any]]:
    """Build train/val dataloaders and optional class-weight tensor.

    Primary val files stay in the kagglehub tree. ``data.val_label_corrections``
    is disabled (null since manifest 1.4.0 — the overlay was circular eval,
    see DATA_SOURCES.md) and must stay null; the machinery remains for audit
    tooling only. Optional ``data.extra_train_sources`` are merged into the
    training set only (same 7 classes).

    Returns ``(train_loader, val_loader, class_weights, weight_info)`` where
    ``weight_info`` holds resolved / auto weights and training source metadata.
    """
    train_dir, val_dir = get_dataset_paths(cfg)
    class_names = cfg["class_names"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    primary_name = str(data_cfg.get("source_name", "primary"))

    train_tf = build_transforms(cfg, train=True)
    val_tf = build_transforms(cfg, train=False)

    primary_train = FERDataset(
        train_dir,
        class_names,
        transform=train_tf,
        source_name=primary_name,
    )
    train_parts: list[FERDataset] = [primary_train]
    source_counts: list[dict[str, Any]] = [
        {"name": primary_name, "role": "primary_train", "num_samples": len(primary_train)}
    ]

    for extra in get_extra_train_sources(cfg):
        extra_ds = FERDataset(
            extra["train_dir"],
            class_names,
            transform=train_tf,
            folder_map=extra.get("folder_map"),
            source_name=str(extra["name"]),
        )
        train_parts.append(extra_ds)
        source_counts.append(
            {
                "name": extra["name"],
                "role": "extra_train_only",
                "kaggle_slug": extra.get("kaggle_slug"),
                "num_samples": len(extra_ds),
            }
        )

    train_ds: Dataset = primary_train if len(train_parts) == 1 else ConcatDataset(train_parts)
    train_targets = _train_targets(train_ds)
    logger.info(
        "Train set total=%d from %d source(s): %s",
        len(train_targets),
        len(train_parts),
        {s["name"]: s["num_samples"] for s in source_counts},
    )

    val_ds = FERDataset(
        val_dir,
        class_names,
        transform=val_tf,
        source_name=f"{primary_name}_val",
    )
    apply_corrections_from_cfg(val_ds, cfg)

    class_weights: torch.Tensor | None = None
    auto_weights: torch.Tensor | None = None
    overrides = train_cfg.get("class_weight_overrides") or {}
    multipliers = train_cfg.get("class_weight_multipliers") or {}
    if train_cfg.get("use_class_weights", True):
        class_weights, auto_weights = compute_class_weights(
            train_targets,
            num_classes=len(class_names),
            class_names=class_names,
            overrides=overrides if overrides else None,
            multipliers=multipliers if multipliers else None,
        )
        logger.info(
            "Class weights (auto): %s",
            class_weights_to_dict(auto_weights, class_names),
        )
        logger.info(
            "Class weights (final): %s",
            class_weights_to_dict(class_weights, class_names),
        )
        if overrides:
            logger.info("Applied class_weight_overrides: %s", dict(overrides))
        if multipliers:
            logger.info("Applied class_weight_multipliers: %s", dict(multipliers))

    weight_info: dict[str, Any] = {
        "use_class_weights": bool(train_cfg.get("use_class_weights", True)),
        "class_weight_overrides": dict(overrides) if overrides else {},
        "class_weight_multipliers": dict(multipliers) if multipliers else {},
        "class_weights_auto": class_weights_to_dict(auto_weights, class_names),
        "class_weights": class_weights_to_dict(class_weights, class_names),
        "training_sources": describe_training_sources(cfg),
        "training_source_counts": source_counts,
        "train_num_samples": len(train_targets),
        "val_num_samples": len(val_ds),
    }

    manifest = load_manifest()
    if manifest is not None:
        weight_info["data_version"] = manifest.get("version")
        weight_info["data_fingerprint"] = manifest.get("fingerprint")
        primary_sha = None
        for src in manifest.get("sources") or []:
            if src.get("role") == "primary_train_val":
                primary_sha = src.get("sha256")
                break
        if primary_sha is None and manifest.get("sources"):
            primary_sha = manifest["sources"][0].get("sha256")
        weight_info["data_manifest_sha256"] = primary_sha
        logger.info(
            "Data version=%s fingerprint=%s",
            weight_info["data_version"],
            (weight_info["data_fingerprint"] or "")[:12] or "n/a",
        )
    else:
        weight_info["data_version"] = None
        weight_info["data_fingerprint"] = None
        weight_info["data_manifest_sha256"] = None
        logger.warning(
            "No data/manifest.json found; run `python -m data.download` to version data."
        )

    sampler = None
    shuffle = True
    if train_cfg.get("use_weighted_sampler", True):
        sampler = build_weighted_sampler(train_targets)
        shuffle = False  # mutually exclusive with sampler

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, class_weights, weight_info


def create_val_loader(cfg: dict[str, Any]) -> DataLoader:
    """Build validation DataLoader only (skips scanning the train split)."""
    _, val_dir = get_dataset_paths(cfg)
    class_names = cfg["class_names"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    val_ds = FERDataset(val_dir, class_names, transform=build_transforms(cfg, train=False))
    apply_corrections_from_cfg(val_ds, cfg)
    return DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )


def create_extra_val_loaders(cfg: dict[str, Any]) -> dict[str, DataLoader]:
    """Build eval-only DataLoaders for extra sources that define ``val_dir``.

    These give a second validation view (e.g. RAF-DB val) alongside the
    primary val split. They are never trained on and never checkpointed on.
    """
    class_names = cfg["class_names"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    val_tf = build_transforms(cfg, train=False)

    loaders: dict[str, DataLoader] = {}
    for extra in get_extra_train_sources(cfg):
        if extra.get("val_dir") is None:
            continue
        ds = FERDataset(
            extra["val_dir"],
            class_names,
            transform=val_tf,
            folder_map=extra.get("folder_map"),
            source_name=f"{extra['name']}_val",
        )
        loaders[str(extra["name"])] = DataLoader(
            ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
        )
        logger.info(
            "Extra val loader '%s': %d samples (eval-only)",
            extra["name"],
            len(ds),
        )
    return loaders
