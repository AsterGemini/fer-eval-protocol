"""Pareto class-F1 gate for RAF-DB val vs a locked baseline.

A candidate may replace best.pt only if every class currently under
``weak_threshold`` strictly improves and every class at or above the
threshold does not drop. FER val accuracy is a separate regression floor.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils import project_root

DEFAULT_BASELINE = "configs/raf_pareto_baseline.json"
DEFAULT_WEAK_THRESHOLD = 0.8


@dataclass
class ParetoResult:
    ok: bool
    weak: list[str]
    strong: list[str]
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "weak": self.weak,
            "strong": self.strong,
            "failures": self.failures,
        }


def split_classes(
    baseline_f1: dict[str, float],
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """Return (weak, strong) class names from the locked baseline, not the candidate."""
    weak = sorted(name for name, f1 in baseline_f1.items() if f1 < weak_threshold)
    strong = sorted(name for name, f1 in baseline_f1.items() if f1 >= weak_threshold)
    return weak, strong


def extract_class_f1(payload: dict[str, Any], split: str = "raf_db") -> dict[str, float]:
    """Pull a class→F1 map from an evaluate report, run log, or flat dict."""
    if "class_f1" in payload and isinstance(payload["class_f1"], dict):
        return {k: float(v) for k, v in payload["class_f1"].items()}

    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        if "class_f1" in metrics and isinstance(metrics["class_f1"], dict):
            return {k: float(v) for k, v in metrics["class_f1"].items()}
        per_class = metrics.get("per_class")
        if isinstance(per_class, dict) and per_class:
            first = next(iter(per_class.values()))
            if isinstance(first, dict) and "f1" in first:
                return {k: float(v["f1"]) for k, v in per_class.items()}

    extra = payload.get("extra_val") or (metrics.get("extra_val") if isinstance(metrics, dict) else None)
    if isinstance(extra, dict) and split in extra:
        return extract_class_f1(extra[split], split=split)

    history = payload.get("history")
    if isinstance(history, dict):
        extra_hist = history.get("extra_val") or []
        if extra_hist:
            best_i = 0
            best_v = float("-inf")
            for i, row in enumerate(extra_hist):
                if not isinstance(row, dict) or split not in row:
                    continue
                v = row[split].get("macro_f1")
                if v is not None and float(v) > best_v:
                    best_v = float(v)
                    best_i = i
            return extract_class_f1(extra_hist[best_i][split], split=split)

    raise ValueError("No class_f1 map found in payload")


def load_baseline(path: str | Path | None = None) -> dict[str, Any]:
    """Load the locked RAF-DB class-F1 baseline JSON."""
    baseline_path = Path(path) if path is not None else project_root() / DEFAULT_BASELINE
    if not baseline_path.is_absolute():
        baseline_path = project_root() / baseline_path
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    if "class_f1" not in data:
        raise ValueError(f"{baseline_path} missing class_f1")
    return data


def evaluate_pareto(
    baseline_f1: dict[str, float],
    candidate_f1: dict[str, float],
    *,
    weak_threshold: float = DEFAULT_WEAK_THRESHOLD,
    fer_acc: float | None = None,
    fer_acc_floor: float | None = None,
) -> ParetoResult:
    """Compare candidate class F1 against a locked baseline.

    Weak classes (baseline F1 < threshold) must strictly increase.
    Strong classes (baseline F1 >= threshold) must not decrease.
    """
    missing = sorted(set(baseline_f1) - set(candidate_f1))
    if missing:
        return ParetoResult(
            ok=False,
            weak=[],
            strong=[],
            failures=[f"candidate missing classes: {missing}"],
        )

    weak, strong = split_classes(baseline_f1, weak_threshold)
    failures: list[str] = []

    for name in weak:
        base = baseline_f1[name]
        cand = float(candidate_f1[name])
        if cand <= base:
            failures.append(
                f"weak {name}: {cand:.4f} does not exceed baseline {base:.4f}"
            )

    for name in strong:
        base = baseline_f1[name]
        cand = float(candidate_f1[name])
        if cand < base:
            failures.append(
                f"strong {name}: {cand:.4f} dropped below baseline {base:.4f}"
            )

    if fer_acc_floor is not None:
        if fer_acc is None:
            failures.append("FER val acc missing; cannot check regression floor")
        elif float(fer_acc) < float(fer_acc_floor):
            failures.append(
                f"FER val acc {float(fer_acc):.4f} < floor {float(fer_acc_floor):.4f}"
            )

    return ParetoResult(ok=not failures, weak=weak, strong=strong, failures=failures)


def evaluate_payload(
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None = None,
    *,
    fer_acc: float | None = None,
) -> ParetoResult:
    """Evaluate a parsed metrics/run JSON against the locked baseline."""
    base = baseline if baseline is not None else load_baseline()
    split = str(base.get("split") or "raf_db")
    candidate_f1 = extract_class_f1(candidate, split=split)
    if fer_acc is None:
        fer_acc = _infer_fer_acc(candidate)
    return evaluate_pareto(
        {k: float(v) for k, v in base["class_f1"].items()},
        candidate_f1,
        weak_threshold=float(base.get("weak_threshold", DEFAULT_WEAK_THRESHOLD)),
        fer_acc=fer_acc,
        fer_acc_floor=base.get("fer_val_accuracy_floor"),
    )


def _infer_fer_acc(payload: dict[str, Any]) -> float | None:
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and payload.get("eval_split") in (None, "fer2013"):
        acc = metrics.get("accuracy")
        if acc is not None and payload.get("eval_split") is None:
            return float(acc)
    history = payload.get("history")
    if isinstance(history, dict) and history.get("val_accuracy"):
        extra = history.get("extra_val") or []
        if extra:
            best_i = 0
            best_v = float("-inf")
            for i, row in enumerate(extra):
                raf = row.get("raf_db") if isinstance(row, dict) else None
                if not raf:
                    continue
                v = raf.get("macro_f1")
                if v is not None and float(v) > best_v:
                    best_v = float(v)
                    best_i = i
            return float(history["val_accuracy"][best_i])
        return float(history["val_accuracy"][-1])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="RAF-DB Pareto class-F1 gate")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", required=True, help="evaluate JSON or run_*.json")
    parser.add_argument("--fer-acc", type=float, default=None)
    args = parser.parse_args()

    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    result = evaluate_payload(
        candidate,
        load_baseline(args.baseline),
        fer_acc=args.fer_acc,
    )
    print(json.dumps(result.as_dict(), indent=2))
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
