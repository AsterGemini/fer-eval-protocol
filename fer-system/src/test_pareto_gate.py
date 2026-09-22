"""Verified tests for the RAF-DB Pareto class-F1 gate.

Run from fer-system/:

    python -m src.test_pareto_gate
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pareto_gate import (  # noqa: E402
    evaluate_pareto,
    evaluate_payload,
    extract_class_f1,
    load_baseline,
    split_classes,
)
from src.utils import load_config  # noqa: E402

# 24-0130 E2 — measured miss vs 22-1038 E2 (run_20260824T013007Z.json).
RUN_B_PEAK = {
    "angry": 0.825,
    "disgust": 0.6503067484662577,
    "fear": 0.7272727272727273,
    "happy": 0.9369527145359019,
    "neutral": 0.8258992805755395,
    "sad": 0.83399209486166,
    "surprise": 0.8802395209580839,
}


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_baseline_splits_disgust_and_fear_as_weak() -> None:
    base = load_baseline()
    f1 = {k: float(v) for k, v in base["class_f1"].items()}
    weak, strong = split_classes(f1, float(base["weak_threshold"]))
    _assert(weak == ["disgust", "fear"], f"weak={weak}")
    _assert(
        strong == ["angry", "happy", "neutral", "sad", "surprise"],
        f"strong={strong}",
    )
    _assert(f1["disgust"] < 0.8, f1["disgust"])
    _assert(f1["fear"] < 0.8, f1["fear"])
    for name in strong:
        _assert(f1[name] >= 0.8, f"{name}={f1[name]}")


def test_identity_is_not_a_win() -> None:
    base = load_baseline()["class_f1"]
    result = evaluate_pareto(base, dict(base), fer_acc=0.7114, fer_acc_floor=0.71)
    _assert(not result.ok, "equal weak F1 must fail the strict-increase rule")
    _assert(any("disgust" in f for f in result.failures), result.failures)
    _assert(any("fear" in f for f in result.failures), result.failures)
    _assert(not any(f.startswith("strong ") for f in result.failures), result.failures)


def test_run_b_peak_fails_pareto() -> None:
    base = load_baseline()["class_f1"]
    result = evaluate_pareto(base, RUN_B_PEAK, fer_acc=0.7087, fer_acc_floor=0.71)
    _assert(not result.ok, result.failures)
    joined = " ".join(result.failures)
    _assert("disgust" in joined, result.failures)
    _assert("fear" in joined, result.failures)
    _assert("angry" in joined, result.failures)
    _assert("FER val acc" in joined, result.failures)


def test_synthetic_pareto_win_passes() -> None:
    base = load_baseline()["class_f1"]
    candidate = dict(base)
    candidate["disgust"] = base["disgust"] + 0.02
    candidate["fear"] = base["fear"] + 0.02
    result = evaluate_pareto(base, candidate, fer_acc=0.7114, fer_acc_floor=0.71)
    _assert(result.ok, result.failures)
    _assert(result.failures == [], result.failures)


def test_strong_class_drop_vetoes_even_if_weak_rise() -> None:
    base = load_baseline()["class_f1"]
    candidate = dict(base)
    candidate["disgust"] = base["disgust"] + 0.05
    candidate["fear"] = base["fear"] + 0.05
    candidate["angry"] = base["angry"] - 0.01
    result = evaluate_pareto(base, candidate, fer_acc=0.72, fer_acc_floor=0.71)
    _assert(not result.ok, result.failures)
    _assert(any("strong angry" in f for f in result.failures), result.failures)
    _assert(not any("disgust" in f for f in result.failures), result.failures)


def test_fer_floor_veto() -> None:
    base = load_baseline()["class_f1"]
    candidate = dict(base)
    candidate["disgust"] = base["disgust"] + 0.02
    candidate["fear"] = base["fear"] + 0.02
    result = evaluate_pareto(base, candidate, fer_acc=0.70, fer_acc_floor=0.71)
    _assert(not result.ok, result.failures)
    _assert(any("FER val acc" in f for f in result.failures), result.failures)


def test_extract_from_evaluate_and_run_shapes() -> None:
    base = load_baseline()
    evaluate_shape = {
        "eval_split": "raf_db",
        "metrics": {
            "per_class": {
                name: {"f1": f1, "support": 10} for name, f1 in base["class_f1"].items()
            }
        },
    }
    run_shape = {
        "history": {
            "val_accuracy": [0.70, 0.7114350410416077],
            "extra_val": [
                {"raf_db": {"macro_f1": 0.80, "class_f1": {k: 0.5 for k in base["class_f1"]}}},
                {"raf_db": {"macro_f1": 0.8170916071503419, "class_f1": base["class_f1"]}},
            ],
        }
    }
    _assert(extract_class_f1(evaluate_shape)["disgust"] == base["class_f1"]["disgust"], "eval shape")
    _assert(extract_class_f1(run_shape)["fear"] == base["class_f1"]["fear"], "run shape")
    identity = evaluate_payload(run_shape, base)
    _assert(not identity.ok, "best epoch of 22-1038 vs itself must fail weak increase")


def test_public_config_keeps_protocol_and_sad_only_weights() -> None:
    cfg = load_config("configs/public.yaml")
    train = cfg["training"]
    _assert(train.get("monitor_metric") == "raf_db.macro_f1", train)
    _assert(
        cfg["data"].get("val_label_corrections") in (None, "", "null"),
        cfg["data"].get("val_label_corrections"),
    )
    multipliers = train.get("class_weight_multipliers") or {}
    _assert("disgust" not in multipliers, multipliers)
    _assert("fear" not in multipliers, multipliers)
    _assert(multipliers.get("sad") == 1.5, multipliers)


def main() -> None:
    test_baseline_splits_disgust_and_fear_as_weak()
    test_identity_is_not_a_win()
    test_run_b_peak_fails_pareto()
    test_synthetic_pareto_win_passes()
    test_strong_class_drop_vetoes_even_if_weak_rise()
    test_fer_floor_veto()
    test_extract_from_evaluate_and_run_shapes()
    test_public_config_keeps_protocol_and_sad_only_weights()
    print(
        "METRIC ok: RAF weak=disgust,fear must rise; "
        "strong classes must not drop; 24-0130 E2 is a known fail"
    )


if __name__ == "__main__":
    main()
