"""timm-backed FER classifier with a replaceable backbone and custom head."""

from __future__ import annotations

import logging
from typing import Any

import timm
import torch
import torch.nn as nn

logger = logging.getLogger("fer.model")


class FERModel(nn.Module):
    """Emotion classifier: pretrained timm backbone + custom 7-class head.

    Design choices
    --------------
    - ``num_classes=0`` strips timm's default classifier so we control the head.
    - Dropout before the linear layers reduces overfitting on FER's modest size.
    - ``drop_path_rate`` is timm stochastic depth on backbone blocks (train-only).
    - Freeze / unfreeze helpers let callers control which backbone params train
      without knowing timm internals.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.drop_path_rate = drop_path_rate

        # Validate name existence only — pretrained availability is handled by timm.
        if backbone not in timm.list_models():
            raise ValueError(
                f"Unknown timm backbone '{backbone}'. "
                f"Examples: efficientnet_b0, efficientnet_b2, resnet50, vit_base_patch16_224"
            )

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # feature extractor only
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        hidden = max(feat_dim // 2, num_classes * 4)

        # Small MLP head — clearer than a single Linear for FER transfer learning.
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )
        logger.info(
            "Built FERModel backbone=%s feat_dim=%d hidden=%d num_classes=%d "
            "pretrained=%s drop_path_rate=%g dropout=%g",
            backbone,
            feat_dim,
            hidden,
            num_classes,
            pretrained,
            drop_path_rate,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        """Freeze the backbone; train only the classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True
        logger.info("Froze backbone; head remains trainable.")

    def unfreeze_backbone(self, last_n_blocks: int | None = None) -> None:
        """Unfreeze entire backbone or only the last N child modules.

        ``last_n_blocks=None`` unfreezes everything (simple default).
        Passing an int unfreezes only the last N *immediate* children of the
        backbone module — a coarse but backbone-agnostic approximation of
        "fine-tune last stages".
        """
        if last_n_blocks is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Unfroze entire backbone.")
            return

        children = list(self.backbone.children())
        if last_n_blocks <= 0 or last_n_blocks > len(children):
            raise ValueError(
                f"last_n_blocks={last_n_blocks} invalid for backbone with {len(children)} children"
            )

        # Freeze all, then unfreeze the last N children.
        for param in self.backbone.parameters():
            param.requires_grad = False
        for child in children[-last_n_blocks:]:
            for param in child.parameters():
                param.requires_grad = True
        logger.info("Unfroze last %d backbone blocks (%d children total).", last_n_blocks, len(children))

    def trainable_parameter_count(self) -> tuple[int, int]:
        """Return (trainable, total) parameter counts."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def build_model(cfg: dict[str, Any]) -> FERModel:
    """Factory: construct FERModel from config."""
    model_cfg = cfg["model"]
    return FERModel(
        backbone=model_cfg["backbone"],
        num_classes=int(model_cfg["num_classes"]),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
    )
