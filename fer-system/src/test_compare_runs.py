"""Tests for flat run compare (agent-facing, no nested history walk).

Run from fer-system/:

    python -m src.test_compare_runs
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.compare_runs import (  # noqa: E402
    REQUIRED_SUMMARY_KEYS,
    build_report,
    compare_pair,
    format_text,
    summarize_run,
)
from src.pareto_gate import load_baseline  # noqa: E402

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _train_payload(
    *,
    stamp: str,
    fer_acc: list[float],
    raf_f1: list[float],
    fear_f1: list[float],
    fear_ce: list[float],
    sad_f1: list[float] | None = None,
    sad_ce: list[float] | None = None,
    lr: float = 1e-4,
    multipliers: dict | None = None,
    raf_class_f1: dict[str, float] | None = None,
) -> dict:
    n = len(fer_acc)
    sad_f1 = sad_f1 or [0.83] * n
    sad_ce = sad_ce or [1.2] * n
    base_f1 = raf_class_f1 or {name: 0.8 for name in CLASS_NAMES}
    extra = []
    val_class_f1 = []
    val_class_loss = []
    for i in range(n):
        class_f1 = dict(base_f1)
        class_f1["fear"] = fear_f1[i]
        class_f1["sad"] = sad_f1[i]
        extra.append(
            {
                "raf_db": {
                    "macro_f1": raf_f1[i],
                    "accuracy": 0.9,
                    "class_f1": class_f1,
                    "class_loss": {"fear": fear_ce[i], "sad": sad_ce[i]},
                }
            }
        )
        val_class_f1.append({"fear": fear_f1[i], "sad": sad_f1[i]})
        val_class_loss.append({"fear": fear_ce[i], "sad": sad_ce[i]})
    return {
        "timestamp_utc": stamp,
        "best_metric_name": "raf_db.macro_f1",
        "best_metric_value": max(raf_f1),
        "hyperparameters": {
            "training": {
                "lr": lr,
                "epochs": n,
                "class_weight_overrides": {},
                "class_weight_multipliers": multipliers or {},
                "label_smoothing": 0.05,
                "monitor_metric": "raf_db.macro_f1",
                "resume_from_best": True,
            },
            "scheduler": {"eta_min_ratio": 1.0},
        },
        "metrics": {
            "prior_best_metric": 0.79,
            "resumed_from": "artifacts/models/efficientnet_b0/checkpoints/best.pt",
            "data_version": "1.5.0",
            "run_stamp": stamp,
        },
        "history": {
            "val_accuracy": fer_acc,
            "val_macro_f1": [a * 0.9 for a in fer_acc],
            "val_class_f1": val_class_f1,
            "val_class_loss": val_class_loss,
            "extra_val": extra,
        },
    }


def test_summary_is_flat_and_complete() -> None:
    payload = _train_payload(
        stamp="20260825T000000Z",
        fer_acc=[0.70, 0.71, 0.72],
        raf_f1=[0.79, 0.80, 0.81],
        fear_f1=[0.70, 0.71, 0.72],
        fear_ce=[1.1, 1.0, 0.9],
    )
    summary = summarize_run(payload, source="logs/run_20260825T000000Z.json")
    missing = REQUIRED_SUMMARY_KEYS - summary.keys()
    _assert(not missing, f"missing {missing}")
    _assert("history" not in summary, "summary must not include nested history")
    _assert(summary["stamp"] == "20260825T000000Z", summary["stamp"])
    _assert(summary["raf_db_macro_f1_best"] == 0.81, summary["raf_db_macro_f1_best"])
    _assert(summary["best_fer_acc"] == 0.72, summary["best_fer_acc"])
    _assert(summary["best_monitor_epoch"] == 3, summary["best_monitor_epoch"])
    _assert(summary["fear_f1_last"] == 0.72, summary["fear_f1_last"])
    _assert(summary["n_epochs"] == 3, summary["n_epochs"])
    dumped = json.dumps(summary)
    _assert("extra_val" not in dumped, "must not leak extra_val arrays")


def test_summary_smaller_than_nested_history() -> None:
    n = 12
    payload = _train_payload(
        stamp="big",
        fer_acc=[0.70 + i * 0.001 for i in range(n)],
        raf_f1=[0.79 + i * 0.001 for i in range(n)],
        fear_f1=[0.70] * n,
        fear_ce=[1.2] * n,
        raf_class_f1={name: 0.8 + (0.001 * i) for i, name in enumerate(CLASS_NAMES)},
    )
    summary = summarize_run(payload)
    src_bytes = len(json.dumps(payload))
    out_bytes = len(json.dumps(summary))
    _assert(src_bytes > 4000, f"fixture too small ({src_bytes}) to prove flattening")
    _assert(out_bytes * 3 < src_bytes, f"summary {out_bytes} vs source {src_bytes}")
    _assert("history" not in summary, "history leaked")


def test_plateau_last3_flat_and_monitor_bounced() -> None:
    payload = _train_payload(
        stamp="plat",
        fer_acc=[0.7000, 0.7100, 0.7110, 0.7112, 0.7111],
        raf_f1=[0.790, 0.797, 0.791, 0.790, 0.789],
        fear_f1=[0.72] * 5,
        fear_ce=[1.2] * 5,
    )
    summary = summarize_run(payload)
    _assert(summary["plateau"] is True, summary)
    _assert(summary["last3_fer_acc_span_pp"] is not None, summary)
    _assert(summary["last3_fer_acc_span_pp"] <= 0.3, summary["last3_fer_acc_span_pp"])
    _assert(summary["climbing"] is False, summary["climbing"])
    _assert(summary["best_monitor_epoch"] == 2, summary["best_monitor_epoch"])


def test_climbing_is_not_plateau() -> None:
    payload = _train_payload(
        stamp="climb",
        fer_acc=[0.70, 0.71, 0.72, 0.73, 0.74],
        raf_f1=[0.79, 0.80, 0.81, 0.82, 0.83],
        fear_f1=[0.72] * 5,
        fear_ce=[1.2] * 5,
    )
    summary = summarize_run(payload)
    _assert(summary["plateau"] is False, summary)
    _assert(summary["climbing"] is True, summary["climbing"])


def test_abort_fear_ce_with_stuck_f1() -> None:
    payload = _train_payload(
        stamp="abort",
        fer_acc=[0.70, 0.71, 0.71],
        raf_f1=[0.79, 0.80, 0.80],
        fear_f1=[0.57, 0.57, 0.57],
        fear_ce=[1.8, 2.2, 2.6],
    )
    summary = summarize_run(payload)
    _assert(summary["abort_classes"] == ["fear"], summary["abort_classes"])
    _assert(summary["fear_ce_last"] == 2.6, summary["fear_ce_last"])


def test_sad_boost_is_not_abort() -> None:
    payload = _train_payload(
        stamp="sad",
        fer_acc=[0.70, 0.71, 0.71],
        raf_f1=[0.79, 0.80, 0.80],
        fear_f1=[0.72, 0.73, 0.73],
        fear_ce=[1.1, 1.1, 1.1],
        sad_f1=[0.83, 0.84, 0.84],
        sad_ce=[1.16, 1.24, 1.31],
        multipliers={"sad": 1.5},
    )
    summary = summarize_run(payload)
    _assert(summary["abort_classes"] == [], summary["abort_classes"])
    _assert(summary["class_weight_multipliers"] == {"sad": 1.5}, summary)


def test_evaluate_shape_does_not_require_history() -> None:
    payload = {
        "checkpoint": "artifacts/models/efficientnet_b0/checkpoints/best.pt",
        "hyperparameters": {
            "training": {
                "lr": 1e-4,
                "epochs": 8,
                "class_weight_overrides": {},
                "class_weight_multipliers": {},
                "label_smoothing": 0.05,
                "monitor_metric": "macro_f1",
                "resume_from_best": False,
            },
            "scheduler": {"eta_min_ratio": 1.0},
        },
        "metrics": {
            "accuracy": 0.7114,
            "macro_f1": 0.68,
            "class_f1": {name: 0.7 for name in CLASS_NAMES},
        },
    }
    summary = summarize_run(payload, source="figures/metrics.json")
    _assert(summary["plateau"] is None, summary["plateau"])
    _assert(summary["final_val_accuracy"] == 0.7114, summary["final_val_accuracy"])
    _assert(summary["n_epochs"] == 0, summary["n_epochs"])


def test_compare_progress_and_hp_delta() -> None:
    prev = summarize_run(
        _train_payload(
            stamp="old",
            fer_acc=[0.70, 0.71],
            raf_f1=[0.790, 0.791],
            fear_f1=[0.70, 0.70],
            fear_ce=[1.2, 1.2],
            lr=1e-4,
        )
    )
    cand = summarize_run(
        _train_payload(
            stamp="new",
            fer_acc=[0.71, 0.722],
            raf_f1=[0.791, 0.810],
            fear_f1=[0.70, 0.73],
            fear_ce=[1.2, 1.1],
            lr=3e-5,
        )
    )
    cmp_ = compare_pair(prev, cand)
    _assert(cmp_["progress"] is True, cmp_)
    _assert(cmp_["new_high_raf"] is True, cmp_)
    _assert(cmp_["new_high_fer"] is True, cmp_)
    _assert("fear" in cmp_["class_progress"], cmp_)
    _assert("lr" in cmp_["hp_changed"], cmp_["hp_changed"])


def test_compare_plateau_is_not_progress() -> None:
    prev = summarize_run(
        _train_payload(
            stamp="old",
            fer_acc=[0.7110, 0.7111, 0.7112],
            raf_f1=[0.797, 0.797, 0.797],
            fear_f1=[0.72] * 3,
            fear_ce=[1.2] * 3,
        )
    )
    cand = summarize_run(
        _train_payload(
            stamp="flat",
            fer_acc=[0.7110, 0.7112, 0.7111],
            raf_f1=[0.797, 0.796, 0.795],
            fear_f1=[0.72] * 3,
            fear_ce=[1.2] * 3,
        )
    )
    cmp_ = compare_pair(prev, cand)
    _assert(cmp_["progress"] is False, cmp_)
    _assert(cmp_["plateau"] is True, cmp_)


def test_report_has_no_history_and_includes_pareto() -> None:
    payload = _train_payload(
        stamp="p",
        fer_acc=[0.7114, 0.7114],
        raf_f1=[0.817, 0.817],
        fear_f1=[0.757, 0.757],
        fear_ce=[1.2, 1.2],
        raf_class_f1={k: float(v) for k, v in load_baseline()["class_f1"].items()},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run_p.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        report = build_report([path])
    dumped = json.dumps(report)
    _assert('"history"' not in dumped, "report leaked history")
    _assert(report["runs"][0]["pareto"] is not None, report["runs"][0])
    _assert(report["runs"][0]["pareto"]["ok"] is False, "identity vs locked baseline must fail")
    _assert(report["all_time"]["raf_db_macro_f1"]["stamp"] == "p", report["all_time"])
    text = format_text(report)
    _assert("raf_best" in text, text)
    _assert("p" in text, text)


def test_agent_ingest_is_the_report_not_the_run_file() -> None:
    """METRIC: required COLAB fields are in report JSON; nested history is not."""
    payload = _train_payload(
        stamp="metric",
        fer_acc=[0.70, 0.71, 0.72, 0.73],
        raf_f1=[0.79, 0.80, 0.81, 0.82],
        fear_f1=[0.70, 0.71, 0.72, 0.73],
        fear_ce=[1.2, 1.1, 1.0, 0.9],
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run_metric.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        report = build_report([path])
    row = report["runs"][0]
    for key in REQUIRED_SUMMARY_KEYS:
        _assert(key in row, key)
    _assert("history" not in row, row.keys())
    _assert("hyperparameters" not in row, "full cfg dump would recreate the nested walk")


def test_cli_json_has_no_nested_history() -> None:
    payload = _train_payload(
        stamp="cli",
        fer_acc=[0.70, 0.71, 0.72],
        raf_f1=[0.79, 0.80, 0.81],
        fear_f1=[0.70, 0.71, 0.72],
        fear_ce=[1.2, 1.1, 1.0],
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run_cli.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-m", "src.compare_runs", str(path)],
            cwd=str(_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    report = json.loads(proc.stdout)
    _assert("history" not in json.dumps(report), proc.stdout[:200])
    _assert(report["runs"][0]["stamp"] == "cli", report["runs"][0]["stamp"])
    _assert(report["compare"] is None, "single run has no pair compare")
    for key in REQUIRED_SUMMARY_KEYS:
        _assert(key in report["runs"][0], key)


def test_checkpoint_hash_match() -> None:
    from src.compare_runs import checkpoint_match, sha256_file

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "best.pt").write_bytes(b"weights-a")
        (d / "run_stampZ.pt").write_bytes(b"weights-a")
        matched = checkpoint_match("stampZ", d)
        _assert(matched is not None and matched["match"] is True, matched)
        (d / "run_stampZ.pt").write_bytes(b"weights-b")
        differed = checkpoint_match("stampZ", d)
        _assert(differed is not None and differed["match"] is False, differed)
        _assert(sha256_file(d / "best.pt") != sha256_file(d / "run_stampZ.pt"), "hashes")


def main() -> None:
    test_summary_is_flat_and_complete()
    test_summary_smaller_than_nested_history()
    test_plateau_last3_flat_and_monitor_bounced()
    test_climbing_is_not_plateau()
    test_abort_fear_ce_with_stuck_f1()
    test_sad_boost_is_not_abort()
    test_evaluate_shape_does_not_require_history()
    test_compare_progress_and_hp_delta()
    test_compare_plateau_is_not_progress()
    test_report_has_no_history_and_includes_pareto()
    test_agent_ingest_is_the_report_not_the_run_file()
    test_cli_json_has_no_nested_history()
    test_checkpoint_hash_match()
    print(
        "METRIC ok: compare_runs stdout/report is flat; "
        "COLAB keys present; history not ingested; plateau/abort/progress gated"
    )


if __name__ == "__main__":
    main()
