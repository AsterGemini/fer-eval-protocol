"""Regression tests for probability-ensemble inference.

Run from fer-system/:

    python -m src.test_inference_ensemble
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference import (  # noqa: E402
    ProbabilityEnsemble,
    _checkpoint_model_config,
    load_predictor,
    predict_pil,
)


class _FixedLogits(nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("_logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._logits.unsqueeze(0).expand(inputs.shape[0], -1)


class _Counting(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls = 0
        self.last_n = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_n = int(inputs.shape[0])
        return self.inner(inputs)


class _IndexedLogits(nn.Module):
    """Return row ``inputs[:, 0, 0, 0]`` from a fixed logit table."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", table)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        idx = inputs[:, 0, 0, 0].long()
        return self.table.index_select(0, idx)


def _peaked_and_uniform_table() -> tuple[torch.Tensor, torch.Tensor]:
    peaked = torch.tensor([8.0, 0.0])
    uniform = torch.tensor([0.0, 0.0])
    b0 = torch.stack([peaked, peaked, uniform, uniform], dim=0)
    b2 = torch.tensor([[0.0, 8.0]]).expand(4, -1).clone()
    return b0, b2


def _indexed_inputs(n: int = 4) -> torch.Tensor:
    inputs = torch.zeros(n, 3, 8, 8)
    for i in range(n):
        inputs[i, 0, 0, 0] = float(i)
    return inputs


def _assert_raises(expected: type[Exception], fn, message: str) -> None:
    try:
        fn()
    except expected:
        return
    raise AssertionError(message)


def _runtime_config() -> dict:
    return {
        "class_names": ["angry", "happy"],
        "data": {
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "model": {
            "backbone": "efficientnet_b0",
            "num_classes": 2,
            "dropout": 0.3,
        },
    }


def _checkpoint_payload(backbone: str = "efficientnet_b2") -> dict:
    cfg = _runtime_config()
    cfg["model"] = {**cfg["model"], "backbone": backbone}
    return {"config": cfg, "backbone": backbone}


def test_weighted_probability_mean_is_exact() -> None:
    first = _FixedLogits([2.0, 0.0])
    second = _FixedLogits([0.0, 2.0])
    ensemble = ProbabilityEnsemble(
        [first, second],
        weights=[1.0, 3.0],
        member_names=["b0", "b2"],
    )

    inputs = torch.zeros(2, 3, 8, 8)
    actual = ensemble(inputs).exp()
    expected = (
        F.softmax(first(inputs), dim=1) * 0.25
        + F.softmax(second(inputs), dim=1) * 0.75
    )

    if not torch.allclose(actual, expected, atol=1e-7):
        raise AssertionError((actual, expected))
    if ensemble.member_names != ["b0", "b2"]:
        raise AssertionError(ensemble.member_names)


def test_prediction_response_reports_ensemble_provenance() -> None:
    ensemble = ProbabilityEnsemble(
        [_FixedLogits([2.0, 0.0]), _FixedLogits([0.0, 2.0])],
        member_names=["b0", "b2"],
    )
    result = predict_pil(
        Image.new("L", (8, 8)),
        ensemble,
        torch.device("cpu"),
        lambda _: torch.zeros(3, 8, 8),
        _runtime_config(),
        top_k=2,
    )

    if set(result["all_probabilities"]) != {"angry", "happy"}:
        raise AssertionError(result)
    if abs(sum(result["all_probabilities"].values()) - 1.0) > 1e-6:
        raise AssertionError(result)
    if result["ensemble"]["members"] != ["b0", "b2"]:
        raise AssertionError(result)


def test_invalid_cascade_options_are_rejected() -> None:
    models = [_FixedLogits([1.0, 0.0]), _FixedLogits([0.0, 1.0])]
    _assert_raises(
        ValueError,
        lambda: ProbabilityEnsemble(models, cascade_enabled=True, cascade_rule="max_prob"),
        "non-entropy cascade rule must fail",
    )
    _assert_raises(
        ValueError,
        lambda: ProbabilityEnsemble(models, cascade_enabled=True, max_entropy=-0.1),
        "negative max_entropy must fail",
    )


def test_confident_batch_does_not_call_b2() -> None:
    first = _FixedLogits([8.0, 0.0])
    second = _Counting(_FixedLogits([0.0, 8.0]))
    ensemble = ProbabilityEnsemble(
        [first, second],
        weights=[0.8, 0.2],
        cascade_enabled=True,
        max_entropy=0.2,
    )
    actual = ensemble(torch.zeros(3, 3, 8, 8)).exp()
    expected = F.softmax(first(torch.zeros(3, 3, 8, 8)), dim=1)
    if second.calls != 0:
        raise AssertionError(second.calls)
    if not torch.allclose(actual, expected, atol=1e-6):
        raise AssertionError((actual, expected))
    stats = ensemble.cascade_stats()
    if stats["n_seen"] != 3 or stats["n_deferred"] != 0 or stats["deferral_rate"] != 0.0:
        raise AssertionError(stats)


def test_uncertain_batch_matches_full_ensemble() -> None:
    first = _FixedLogits([0.0, 0.0])
    second_inner = _FixedLogits([0.0, 8.0])
    second = _Counting(second_inner)
    inputs = torch.zeros(2, 3, 8, 8)
    ensemble = ProbabilityEnsemble(
        [first, second],
        weights=[0.8, 0.2],
        cascade_enabled=True,
        max_entropy=0.2,
    )
    actual = ensemble(inputs).exp()
    expected = (
        F.softmax(first(inputs), dim=1) * 0.8
        + F.softmax(second_inner(inputs), dim=1) * 0.2
    )
    if second.calls != 1 or second.last_n != 2:
        raise AssertionError((second.calls, second.last_n))
    if not torch.allclose(actual, expected, atol=1e-6):
        raise AssertionError((actual, expected))
    if ensemble.cascade_stats()["n_deferred"] != 2:
        raise AssertionError(ensemble.cascade_stats())


def test_mixed_batch_forwards_only_uncertain_rows() -> None:
    b0_table, b2_table = _peaked_and_uniform_table()
    first = _IndexedLogits(b0_table)
    second_inner = _IndexedLogits(b2_table)
    second = _Counting(second_inner)
    inputs = _indexed_inputs(4)
    ensemble = ProbabilityEnsemble(
        [first, second],
        weights=[0.8, 0.2],
        cascade_enabled=True,
        max_entropy=0.2,
    )
    actual = ensemble(inputs).exp()
    if second.calls != 1 or second.last_n != 2:
        raise AssertionError((second.calls, second.last_n))

    peaked = F.softmax(b0_table[0], dim=0)
    mixed = F.softmax(b0_table[2], dim=0) * 0.8 + F.softmax(b2_table[2], dim=0) * 0.2
    if not torch.allclose(actual[0], peaked, atol=1e-6):
        raise AssertionError(actual[0])
    if not torch.allclose(actual[1], peaked, atol=1e-6):
        raise AssertionError(actual[1])
    if not torch.allclose(actual[2], mixed, atol=1e-6):
        raise AssertionError((actual[2], mixed))
    if not torch.allclose(actual[3], mixed, atol=1e-6):
        raise AssertionError((actual[3], mixed))
    stats = ensemble.cascade_stats()
    if stats["n_seen"] != 4 or stats["n_deferred"] != 2:
        raise AssertionError(stats)
    if abs(float(stats["deferral_rate"]) - 0.5) > 1e-9:
        raise AssertionError(stats)


def test_prediction_response_reports_cascade_provenance() -> None:
    ensemble = ProbabilityEnsemble(
        [_FixedLogits([8.0, 0.0]), _FixedLogits([0.0, 8.0])],
        member_names=["b0", "b2"],
        cascade_enabled=True,
        max_entropy=0.2,
    )
    result = predict_pil(
        Image.new("L", (8, 8)),
        ensemble,
        torch.device("cpu"),
        lambda _: torch.zeros(3, 8, 8),
        _runtime_config(),
        top_k=2,
    )
    if result["cascade"]["deferred"] is not False:
        raise AssertionError(result["cascade"])
    if result["cascade"]["rule"] != "entropy":
        raise AssertionError(result["cascade"])
    if result["cascade"]["entropy"] is None or result["cascade"]["entropy"] > 0.2:
        raise AssertionError(result["cascade"])


def test_invalid_weights_are_rejected() -> None:
    models = [_FixedLogits([1.0, 0.0]), _FixedLogits([0.0, 1.0])]
    _assert_raises(
        ValueError,
        lambda: ProbabilityEnsemble(models, weights=[1.0]),
        "weight/member count mismatch must fail",
    )
    _assert_raises(
        ValueError,
        lambda: ProbabilityEnsemble(models, weights=[1.0, 0.0]),
        "zero weights must fail",
    )
    _assert_raises(
        ValueError,
        lambda: load_predictor(
            _runtime_config(),
            "missing-checkpoint.pt",
            weights=[0.0],
        ),
        "invalid weights must fail before checkpoint loading",
    )


def test_checkpoint_metadata_selects_b2_without_pretrained_download() -> None:
    model_cfg = _checkpoint_model_config(
        _runtime_config(),
        _checkpoint_payload(),
        "future-b2.pt",
    )
    if model_cfg["backbone"] != "efficientnet_b2":
        raise AssertionError(model_cfg)
    if model_cfg["pretrained"] is not False:
        raise AssertionError(model_cfg)
    if model_cfg["num_classes"] != 2:
        raise AssertionError(model_cfg)


def test_class_order_mismatch_is_rejected() -> None:
    payload = _checkpoint_payload()
    payload["config"]["class_names"] = ["happy", "angry"]
    _assert_raises(
        ValueError,
        lambda: _checkpoint_model_config(
            _runtime_config(),
            payload,
            "wrong-classes.pt",
        ),
        "different class order must fail",
    )


def test_preprocessing_mismatch_is_rejected() -> None:
    payload = _checkpoint_payload()
    payload["config"]["data"]["image_size"] = 260
    _assert_raises(
        ValueError,
        lambda: _checkpoint_model_config(
            _runtime_config(),
            payload,
            "wrong-size.pt",
        ),
        "different preprocessing must fail",
    )


def test_ensemble_requires_checkpoint_contract_metadata() -> None:
    _assert_raises(
        ValueError,
        lambda: _checkpoint_model_config(
            _runtime_config(),
            {},
            "metadata-free.pt",
            require_contract=True,
        ),
        "metadata-free checkpoints must not enter a heterogeneous ensemble",
    )


def main() -> None:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print("METRIC ok: weighted probability mean, entropy cascade, and checkpoint contracts are verified")


if __name__ == "__main__":
    main()
