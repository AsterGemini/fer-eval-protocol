"""Train / validation image transforms for FER.

Design notes
------------
- Source images are 48×48 grayscale. We convert to RGB (3-channel replicate)
  and resize to 224×224 so ImageNet-pretrained timm backbones apply cleanly.
- Augmentations stay face-aware: modest rotation/affine + erase, not heavy warps
  that would destroy expression cues.
- Hue / saturation jitter are intentionally omitted: after Grayscale(3),
  R=G=B so those ColorJitter axes are identity no-ops.
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms


def build_transforms(cfg: dict[str, Any], train: bool = True) -> transforms.Compose:
    """Build a torchvision transform pipeline from config."""
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})
    image_size = int(data_cfg["image_size"])
    mean = data_cfg["mean"]
    std = data_cfg["std"]

    # Always convert grayscale → RGB before backbone-facing ops.
    ops: list[Any] = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
    ]

    if train:
        jitter = aug_cfg.get("color_jitter", {}) or {}
        affine = aug_cfg.get("affine", {}) or {}
        erase = aug_cfg.get("random_erasing", {}) or {}

        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=float(aug_cfg.get("horizontal_flip_p", 0.5))),
                transforms.RandomRotation(degrees=float(aug_cfg.get("rotation_degrees", 20))),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(
                        float(affine.get("translate", 0.08)),
                        float(affine.get("translate", 0.08)),
                    ),
                    scale=(
                        float(affine.get("scale_min", 0.9)),
                        float(affine.get("scale_max", 1.1)),
                    ),
                ),
                transforms.ColorJitter(
                    brightness=float(jitter.get("brightness", 0.3)),
                    contrast=float(jitter.get("contrast", 0.3)),
                ),
            ]
        )

        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        erase_p = float(erase.get("p", 0.0))
        if erase_p > 0.0:
            scale = erase.get("scale", [0.02, 0.15])
            ratio = erase.get("ratio", [0.3, 3.3])
            ops.append(
                transforms.RandomErasing(
                    p=erase_p,
                    scale=(float(scale[0]), float(scale[1])),
                    ratio=(float(ratio[0]), float(ratio[1])),
                )
            )
        return transforms.Compose(ops)

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(ops)


def build_inference_transform(cfg: dict[str, Any]) -> transforms.Compose:
    """Deterministic transform for single-image inference (same as val)."""
    return build_transforms(cfg, train=False)
