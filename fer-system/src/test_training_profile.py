"""Regression tests for the next partial-unfreeze training profile.

Run from fer-system/:

    python -m src.test_training_profile
"""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.download import MANIFEST_VERSION  # noqa: E402
from src.model import FERModel  # noqa: E402
from src.train import (  # noqa: E402
    configure_backbone_trainability,
    resolve_model_selection,
)
from src.utils import load_config  # noqa: E402


def _assert(cond: bool, msg: object) -> None:
    if not cond:
        raise AssertionError(msg)


def _public_cfg() -> dict:
    return load_config("configs/public.yaml")


def _with_last4(cfg: dict) -> dict:
    """Last-4 is a supported freeze profile, not the public default recipe."""
    resolved = deepcopy(cfg)
    resolved["training"]["freeze_backbone"] = False
    resolved["training"]["unfreeze_last_n_blocks"] = 4
    return resolved


def test_default_config_is_public_recipe() -> None:
    cfg = _public_cfg()
    source_names = [
        str(source.get("name"))
        for source in cfg["data"].get("extra_train_sources") or []
    ]
    _assert(
        source_names
        == [
            "raf_db",
            "affectnet",
            "expw",
            "ferplus_minority",
            "ckplus",
            "minority_boost",
        ],
        source_names,
    )
    extras = {str(s.get("name")): s for s in cfg["data"].get("extra_train_sources") or []}
    _assert(
        extras["expw"].get("kaggle_slug")
        == "shahzadabbas/expression-in-the-wild-expw-dataset",
        extras["expw"],
    )
    _assert("AsterGemini" not in json.dumps(cfg), cfg["data"])

    train = cfg["training"]
    _assert(train.get("monitor_metric") == "raf_db.macro_f1", train)
    _assert(
        cfg["data"].get("val_label_corrections") in (None, "", "null"),
        cfg["data"].get("val_label_corrections"),
    )
    _assert(train.get("resume_from_best") is True, train)
    _assert(train.get("class_weight_multipliers") == {"sad": 1.5}, train)
    _assert(
        Path(cfg["paths"]["best_checkpoint"])
        == Path("artifacts/models/efficientnet_b0/checkpoints/best.pt"),
        cfg["paths"],
    )


def test_committed_configs_have_no_private_expw_slug() -> None:
    for path in Path("configs").glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        _assert("AsterGemini/expw" not in text, path)
        _assert("expw-face-crops" not in text, path)


def test_agent_docs_do_not_claim_shipped_weights() -> None:
    """COLAB / Vast playbooks (and .cursor rules if present) must not imply committed weights."""
    repo = _ROOT.parent
    paths = [
        _ROOT / "COLAB_ARTIFACTS.md",
        _ROOT / "VAST_ARTIFACTS.md",
        _ROOT / "PROTOCOL.md",
        repo / ".cursor" / "rules" / "colab-artifacts.mdc",
        repo / ".cursor" / "rules" / "vast-artifacts.mdc",
        repo / ".cursor" / "skills" / "colab-artifact-import" / "SKILL.md",
        repo / ".cursor" / "skills" / "vast-artifact-import" / "SKILL.md",
    ]
    banned = (
        "tracked b2",
        "git-tracked",
        "tracked efficientnet-b2",
        "except the tracked",
        "clone-and-demo exception",
        "gitignored except",
    )
    for path in paths:
        if not path.is_file():
            continue  # public dump omits private .cursor playbooks
        text = path.read_text(encoding="utf-8").lower()
        for needle in banned:
            _assert(needle not in text, f"{path}: leftover shipping-weights wording {needle!r}")


def test_train_sh_defaults_to_public_yaml() -> None:
    text = Path("scripts/train.sh").read_text(encoding="utf-8")
    _assert('CONFIG="configs/public.yaml"' in text, text)
    _assert('CONFIG="configs/base.yaml"' not in text, text)


def test_manifest_version_matches_public_mix() -> None:
    _assert(MANIFEST_VERSION == "1.4.0", MANIFEST_VERSION)
    manifest = json.loads(Path("data/manifest.json").read_text(encoding="utf-8"))
    # Committed SHA snapshot from maintainer `data.download` against public.yaml.
    _assert(manifest.get("version") == "1.4.0", manifest.get("version"))
    manifest_names = [str(source.get("name")) for source in manifest.get("sources") or []]
    _assert(
        manifest_names
        == [
            "fer2013_jonathanoheix",
            "raf_db",
            "affectnet",
            "expw",
            "ferplus_minority",
            "ckplus",
            "minority_boost",
        ],
        manifest_names,
    )


def test_partial_profile_is_between_head_only_and_full_model() -> None:
    model = FERModel(pretrained=False)

    model.freeze_backbone()
    head_only, total = model.trainable_parameter_count()

    phase = configure_backbone_trainability(
        model, {"freeze_backbone": False, "unfreeze_last_n_blocks": 4}
    )
    partial, partial_total = model.trainable_parameter_count()
    _assert(phase == "partial_last_4", phase)
    _assert(partial_total == total, (partial_total, total))
    _assert(head_only < partial < total, (head_only, partial, total))
    _assert(partial == 1_236_487, partial)

    children = dict(model.backbone.named_children())
    _assert(
        not any(param.requires_grad for param in children["blocks"].parameters()),
        "feature blocks must remain frozen for the N=4 profile",
    )
    _assert(
        all(param.requires_grad for param in children["conv_head"].parameters()),
        "conv_head must be trainable",
    )
    _assert(
        all(param.requires_grad for param in children["bn2"].parameters()),
        "bn2 must be trainable",
    )
    _assert(
        all(param.requires_grad for param in model.head.parameters()),
        "classifier head must be trainable",
    )

    model.unfreeze_backbone(last_n_blocks=None)
    full, full_total = model.trainable_parameter_count()
    _assert(full_total == total, (full_total, total))
    _assert(partial < full == total, (partial, full, total))


def test_conflicting_freeze_profile_is_rejected() -> None:
    model = FERModel(pretrained=False)
    try:
        configure_backbone_trainability(
            model,
            {"freeze_backbone": True, "unfreeze_last_n_blocks": 4},
        )
    except ValueError as exc:
        _assert("conflicts" in str(exc), exc)
    else:
        raise AssertionError("conflicting freeze settings must fail before training")


def test_cli_model_selection_isolates_b1_from_b0() -> None:
    base = _public_cfg()
    b0 = resolve_model_selection(base, model_alias="b0")
    b1 = resolve_model_selection(base, model_alias="b1")

    _assert(b0["model"]["backbone"] == "efficientnet_b0", b0["model"])
    _assert(
        Path(b0["paths"]["best_checkpoint"])
        == Path("artifacts/models/efficientnet_b0/checkpoints/best.pt"),
        b0["paths"],
    )
    _assert(b0["training"].get("pareto_gate") is not True, b0["training"])

    _assert(b1["model"]["backbone"] == "efficientnet_b1", b1["model"])
    _assert(b1["model"]["pretrained"] is True, b1["model"])
    _assert(b1["training"]["resume_from_best"] is True, b1["training"])
    _assert(b1["training"]["pareto_gate"] is False, b1["training"])
    _assert(
        Path(b1["paths"]["best_checkpoint"])
        == Path("artifacts/models/efficientnet_b1/checkpoints/best.pt"),
        b1["paths"],
    )
    _assert(
        b1["paths"]["best_checkpoint"] != b0["paths"]["best_checkpoint"],
        (b0["paths"], b1["paths"]),
    )


def test_b0_dry_run_cli_reports_model_subdir() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "src.train", "--model", "b0", "--dry-run"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(proc.stdout)
    _assert(summary["backbone"] == "efficientnet_b0", summary)
    _assert(summary["pareto_gate"] is False, summary)
    _assert(
        Path(summary["paths"]["best_checkpoint"])
        == Path("artifacts/models/efficientnet_b0/checkpoints/best.pt"),
        summary["paths"],
    )
    _assert(
        Path(summary["paths"]["log_dir"])
        == Path("artifacts/models/efficientnet_b0/logs"),
        summary["paths"],
    )


def test_b1_builds_and_uses_partial_transfer_profile() -> None:
    cfg = resolve_model_selection(_with_last4(_public_cfg()), model_alias="b1")
    model = FERModel(backbone=cfg["model"]["backbone"], pretrained=False)
    phase = configure_backbone_trainability(model, cfg["training"])
    trainable, total = model.trainable_parameter_count()
    _assert(model.backbone_name == "efficientnet_b1", model.backbone_name)
    _assert(phase == "partial_last_4", phase)
    _assert(0 < trainable < total, (trainable, total))
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 224, 224))
    _assert(tuple(logits.shape) == (1, 7), tuple(logits.shape))


def test_b2_last4_leaves_feature_blocks_frozen() -> None:
    """last-4 is conv_head+bn2+head; `blocks` (most of B2) stay ImageNet."""
    cfg = resolve_model_selection(_with_last4(_public_cfg()), model_alias="b2")
    model = FERModel(backbone=cfg["model"]["backbone"], pretrained=False)
    phase = configure_backbone_trainability(model, cfg["training"])
    trainable, total = model.trainable_parameter_count()
    children = dict(model.backbone.named_children())

    _assert(phase == "partial_last_4", phase)
    _assert(trainable < 0.20 * total, (trainable, total))
    _assert(
        not any(param.requires_grad for param in children["blocks"].parameters()),
        "B2 last-4 must leave EfficientNet blocks frozen",
    )
    _assert(
        all(param.requires_grad for param in children["conv_head"].parameters()),
        "conv_head must be trainable",
    )

    model.unfreeze_backbone(last_n_blocks=5)
    n5, total5 = model.trainable_parameter_count()
    _assert(n5 > 0.99 * total5, (n5, total5))
    _assert(
        any(param.requires_grad for param in children["blocks"].parameters()),
        "last_n=5 unfreezes all blocks at once — not a small step",
    )

    model.unfreeze_backbone(last_n_blocks=None)
    full, full_total = model.trainable_parameter_count()
    _assert(full == full_total, (full, full_total))
    _assert(full - trainable > 0.80 * total, (full, trainable, total))


def test_b2_full_backbone_dry_run_clears_last_n() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--model",
            "b2",
            "--epochs",
            "200",
            "--patience",
            "16",
            "--full-backbone",
            "--dry-run",
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(proc.stdout)
    _assert(summary["backbone"] == "efficientnet_b2", summary)
    _assert(summary["unfreeze_last_n_blocks"] is None, summary)
    _assert(summary["freeze_phase"] == "full", summary)
    _assert(summary["epochs"] == 200, summary)
    _assert(summary["early_stopping_patience"] == 16, summary)
    _assert(summary["pareto_gate"] is False, summary)
    _assert(
        Path(summary["paths"]["best_checkpoint"])
        == Path("artifacts/models/efficientnet_b2/checkpoints/best.pt"),
        summary["paths"],
    )


def test_cli_epochs_override_is_in_dry_run() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--model",
            "b2",
            "--epochs",
            "200",
            "--patience",
            "16",
            "--dry-run",
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(proc.stdout)
    _assert(summary["backbone"] == "efficientnet_b2", summary)
    _assert(summary["epochs"] == 200, summary)
    _assert(summary["early_stopping_patience"] == 16, summary)
    _assert(summary["pareto_gate"] is False, summary)
    _assert(summary["unfreeze_last_n_blocks"] is None, summary)
    _assert(summary["freeze_phase"] == "full", summary)


def test_b1_dry_run_cli_reports_isolated_checkpoint() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "src.train", "--model", "b1", "--dry-run"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(proc.stdout)
    _assert(summary["backbone"] == "efficientnet_b1", summary)
    _assert(summary["pretrained"] is True, summary)
    _assert(summary["pareto_gate"] is False, summary)
    _assert(
        Path(summary["paths"]["best_checkpoint"])
        == Path("artifacts/models/efficientnet_b1/checkpoints/best.pt"),
        summary,
    )


def main() -> None:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(
        "METRIC ok: default=public.yaml sources + protocol; "
        "last-4 remains a supported freeze profile; "
        "--full-backbone is freeze_phase=full"
    )


if __name__ == "__main__":
    main()
