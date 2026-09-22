"""Regression tests for Gradio ensemble configuration resolution."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path).resolve() != _APP_DIR]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.app import (  # noqa: E402
    BYO_CHECKPOINT_HINT,
    resolve_cascade_spec,
    resolve_predictor_spec,
    select_existing_predictor_spec,
    _demo_description,
)
from src.utils import load_config  # noqa: E402

_ENV_KEYS = (
    "FER_CHECKPOINT",
    "FER_CHECKPOINTS",
    "FER_ENSEMBLE_WEIGHTS",
    "FER_CASCADE_MAX_ENTROPY",
)


@contextmanager
def _predictor_env(**values: str):
    previous = {key: os.environ.get(key) for key in _ENV_KEYS}
    try:
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _config(enabled: bool = False, cascade: bool = False) -> dict:
    cfg = {
        "paths": {"best_checkpoint": "b0.pt"},
        "inference": {
            "ensemble": {
                "enabled": enabled,
                "method": "weighted_mean_probabilities",
                "members": [
                    {"checkpoint": "b0.pt", "weight": 1.0},
                    {"checkpoint": "b2.pt", "weight": 2.0},
                ],
            }
        },
    }
    if cascade:
        cfg["inference"]["ensemble"]["cascade"] = {
            "enabled": True,
            "rule": "entropy",
            "max_entropy": 0.8,
        }
    return cfg


def test_disabled_config_preserves_single_model_default() -> None:
    with _predictor_env():
        checkpoints, weights = resolve_predictor_spec(_config(enabled=False))
    if checkpoints != ["b0.pt"] or weights is not None:
        raise AssertionError((checkpoints, weights))


def test_enabled_config_resolves_members_and_weights() -> None:
    with _predictor_env():
        checkpoints, weights = resolve_predictor_spec(_config(enabled=True))
    if checkpoints != ["b0.pt", "b2.pt"] or weights != [1.0, 2.0]:
        raise AssertionError((checkpoints, weights))


def test_cli_ensemble_flag_uses_yaml_members_when_disabled() -> None:
    with _predictor_env():
        checkpoints, weights = resolve_predictor_spec(
            _config(enabled=False), ensemble=True
        )
    if checkpoints != ["b0.pt", "b2.pt"] or weights != [1.0, 2.0]:
        raise AssertionError((checkpoints, weights))


def test_cli_ensemble_flag_does_not_override_env() -> None:
    with _predictor_env(FER_CHECKPOINT="env-only.pt"):
        checkpoints, weights = resolve_predictor_spec(
            _config(enabled=False), ensemble=True
        )
    if checkpoints != ["env-only.pt"] or weights is not None:
        raise AssertionError((checkpoints, weights))


def test_plural_environment_override_has_highest_priority() -> None:
    with _predictor_env(
        FER_CHECKPOINT="ignored.pt",
        FER_CHECKPOINTS="env-b0.pt, env-b2.pt",
        FER_ENSEMBLE_WEIGHTS="3,1",
    ):
        checkpoints, weights = resolve_predictor_spec(_config(enabled=True))
    if checkpoints != ["env-b0.pt", "env-b2.pt"] or weights != [3.0, 1.0]:
        raise AssertionError((checkpoints, weights))


def test_public_yaml_ensemble_stays_opt_in() -> None:
    cfg = load_config("configs/public.yaml")
    ensemble = (cfg.get("inference") or {}).get("ensemble") or {}
    if bool(ensemble.get("enabled", False)):
        raise AssertionError(ensemble)
    with _predictor_env():
        checkpoints, weights = resolve_predictor_spec(cfg)
    expected = [cfg["paths"]["best_checkpoint"]]
    if checkpoints != expected or weights is not None:
        raise AssertionError((checkpoints, weights))
    with _predictor_env():
        flagged, flagged_weights = resolve_predictor_spec(cfg, ensemble=True)
    members = ensemble.get("members") or []
    want_ckpts = [str(member["checkpoint"]) for member in members]
    want_weights = [float(member.get("weight", 1.0)) for member in members]
    if flagged != want_ckpts or flagged_weights != want_weights:
        raise AssertionError((flagged, flagged_weights))
    cascade = ensemble.get("cascade") or {}
    if cascade.get("rule") != "entropy":
        raise AssertionError(cascade)
    if bool(cascade.get("enabled", False)) is not True:
        raise AssertionError(cascade)
    if float(cascade.get("max_entropy")) != 0.8:
        raise AssertionError(cascade)


def test_cascade_spec_reads_nested_config() -> None:
    with _predictor_env():
        enabled, max_entropy = resolve_cascade_spec(_config(cascade=True))
    if enabled is not True or max_entropy != 0.8:
        raise AssertionError((enabled, max_entropy))
    with _predictor_env():
        enabled, max_entropy = resolve_cascade_spec(_config(cascade=False))
    if enabled is not False or max_entropy is not None:
        raise AssertionError((enabled, max_entropy))


def test_cascade_env_overrides_config() -> None:
    with _predictor_env(FER_CASCADE_MAX_ENTROPY="1.25"):
        enabled, max_entropy = resolve_cascade_spec(_config(cascade=False))
    if enabled is not True or max_entropy != 1.25:
        raise AssertionError((enabled, max_entropy))


def test_cli_checkpoint_overrides_yaml() -> None:
    with _predictor_env():
        checkpoints, weights = resolve_predictor_spec(
            _config(), extra_checkpoints=["byo.pt"]
        )
    if checkpoints != ["byo.pt"] or weights is not None:
        raise AssertionError((checkpoints, weights))


def test_env_checkpoint_wins_over_cli() -> None:
    with _predictor_env(FER_CHECKPOINT="env-only.pt"):
        checkpoints, weights = resolve_predictor_spec(
            _config(), extra_checkpoints=["cli.pt"]
        )
    if checkpoints != ["env-only.pt"] or weights is not None:
        raise AssertionError((checkpoints, weights))


def test_picker_uses_yaml_when_present() -> None:
    with _predictor_env():
        checkpoints, weights = select_existing_predictor_spec(
            _config(), is_file=lambda path: path == "b0.pt"
        )
    if checkpoints != ["b0.pt"] or weights is not None:
        raise AssertionError((checkpoints, weights))


def test_picker_exits_when_yaml_checkpoint_missing() -> None:
    with _predictor_env():
        try:
            select_existing_predictor_spec(_config(), is_file=lambda path: False)
        except SystemExit as exc:
            message = str(exc)
        else:
            raise AssertionError("expected SystemExit")
    if "b0.pt" not in message:
        raise AssertionError(message)
    if BYO_CHECKPOINT_HINT not in message:
        raise AssertionError(message)
    if "falling back" in message.lower():
        raise AssertionError(message)


def test_picker_cli_checkpoint_does_not_fall_back() -> None:
    with _predictor_env():
        try:
            select_existing_predictor_spec(
                _config(),
                extra_checkpoints=["byo-missing.pt"],
                is_file=lambda path: path == "b0.pt",
            )
        except SystemExit as exc:
            message = str(exc)
        else:
            raise AssertionError("expected SystemExit")
    if "byo-missing.pt" not in message:
        raise AssertionError(message)
    if BYO_CHECKPOINT_HINT not in message:
        raise AssertionError(message)


def test_picker_ensemble_does_not_substitute_another_member() -> None:
    with _predictor_env():
        try:
            select_existing_predictor_spec(
                _config(),
                ensemble=True,
                is_file=lambda path: path == "b2.pt",
            )
        except SystemExit as exc:
            message = str(exc)
        else:
            raise AssertionError("expected SystemExit")
    if "b0.pt" not in message:
        raise AssertionError(message)
    if BYO_CHECKPOINT_HINT not in message:
        raise AssertionError(message)


def test_picker_env_override_does_not_substitute_yaml() -> None:
    with _predictor_env(FER_CHECKPOINT="missing.pt"):
        try:
            select_existing_predictor_spec(
                _config(),
                is_file=lambda path: path == "b0.pt",
            )
        except SystemExit as exc:
            message = str(exc)
        else:
            raise AssertionError("expected SystemExit")
    if "missing.pt" not in message:
        raise AssertionError(message)
    if BYO_CHECKPOINT_HINT not in message:
        raise AssertionError(message)


def test_picker_keeps_public_yaml_path_string_when_b0_exists() -> None:
    cfg = load_config("configs/public.yaml")
    yaml_path = cfg["paths"]["best_checkpoint"]
    with _predictor_env():
        checkpoints, weights = select_existing_predictor_spec(
            cfg, is_file=lambda path: path == yaml_path
        )
    if checkpoints != [yaml_path] or weights is not None:
        raise AssertionError((checkpoints, weights))


def test_demo_description_does_not_attribute_b0_f1_to_b2() -> None:
    b2_path = "artifacts/models/efficientnet_b2/checkpoints/best.pt"
    b2_text = _demo_description([b2_path])
    if "81.71%" in b2_text:
        raise AssertionError(b2_text)
    if b2_path not in b2_text:
        raise AssertionError(b2_text)
    b0_text = _demo_description(
        ["artifacts/models/efficientnet_b0/checkpoints/best.pt"]
    )
    if "81.71%" not in b0_text:
        raise AssertionError(b0_text)


def main() -> None:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print("METRIC ok: app ensemble configuration and override priority are verified")


if __name__ == "__main__":
    main()
