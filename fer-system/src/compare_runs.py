"""Flatten nested run JSON so compare does not walk history by hand.

Canonical store stays ``run_*.json`` / ``metrics.json``. This module is a
view: one compact payload covering COLAB_ARTIFACTS.md §2–3 (hyperparams,
peaks, fear/sad, plateau, abort, Pareto).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from src.pareto_gate import evaluate_payload
from src.utils import project_root

# COLAB_ARTIFACTS.md §3
TARGET_CLASSES = ("fear", "sad")
FER_PLATEAU_SPAN_PP = 0.3
CLASS_F1_PROGRESS = 0.02  # ≥ ~2 pp
ABORT_CE = 2.5
ABORT_F1_LO = 0.56
ABORT_F1_HI = 0.58

REQUIRED_SUMMARY_KEYS = frozenset(
    {
        "stamp",
        "source",
        "lr",
        "epochs",
        "class_weight_overrides",
        "class_weight_multipliers",
        "label_smoothing",
        "monitor_metric",
        "resume_from_best",
        "eta_min_ratio",
        "prior_best_metric",
        "resumed_from",
        "data_version",
        "final_val_accuracy",
        "final_val_macro_f1",
        "best_fer_acc",
        "best_fer_acc_epoch",
        "best_monitor",
        "best_monitor_epoch",
        "raf_db_macro_f1_best",
        "raf_db_macro_f1_last",
        "n_epochs",
        "last3_fer_acc_span_pp",
        "plateau",
        "climbing",
        "fear_f1_last",
        "fear_f1_best",
        "fear_ce_last",
        "fear_ce_max",
        "sad_f1_last",
        "sad_f1_best",
        "sad_ce_last",
        "sad_ce_max",
        "abort_classes",
    }
)

DEFAULT_LOG_DIRS = (
    "artifacts/models/efficientnet_b0/logs",
    "artifacts_from_googlecolab/logs",
)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _last(values: list[Any]) -> Any:
    return values[-1] if values else None


def _best_with_epoch(values: list[float | None]) -> tuple[float | None, int | None]:
    best: float | None = None
    epoch: int | None = None
    for i, raw in enumerate(values, start=1):
        v = _as_float(raw)
        if v is None:
            continue
        if best is None or v > best:
            best = v
            epoch = i
    return best, epoch


def _span_pp(values: list[float | None]) -> float | None:
    nums = [_as_float(v) for v in values]
    nums = [v for v in nums if v is not None]
    if len(nums) < 2:
        return None
    return (max(nums) - min(nums)) * 100.0


def _hp(payload: dict[str, Any]) -> dict[str, Any]:
    hp = payload.get("hyperparameters")
    return hp if isinstance(hp, dict) else {}


def _metrics(payload: dict[str, Any]) -> dict[str, Any]:
    m = payload.get("metrics")
    return m if isinstance(m, dict) else {}


def _history(payload: dict[str, Any]) -> dict[str, Any]:
    h = payload.get("history")
    return h if isinstance(h, dict) else {}


def _class_series(history: dict[str, Any], field: str, class_name: str) -> list[float | None]:
    rows = history.get(field) or []
    out: list[float | None] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(_as_float(row.get(class_name)))
        else:
            out.append(None)
    return out


def _extra_series(history: dict[str, Any], split: str, key: str) -> list[float | None]:
    rows = history.get("extra_val") or []
    out: list[float | None] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get(split), dict):
            out.append(_as_float(row[split].get(key)))
        else:
            out.append(None)
    return out


def _extra_class_series(
    history: dict[str, Any], split: str, field: str, class_name: str
) -> list[float | None]:
    rows = history.get("extra_val") or []
    out: list[float | None] = []
    for row in rows:
        bucket = row.get(split) if isinstance(row, dict) else None
        mapping = bucket.get(field) if isinstance(bucket, dict) else None
        if isinstance(mapping, dict):
            out.append(_as_float(mapping.get(class_name)))
        else:
            out.append(None)
    return out


def _monitor_series(payload: dict[str, Any], monitor: str) -> list[float | None]:
    history = _history(payload)
    if "." in monitor:
        split, key = monitor.split(".", 1)
        return _extra_series(history, split, key)
    key = monitor if monitor.startswith("val_") else f"val_{monitor}"
    rows = history.get(key) or []
    return [_as_float(v) for v in rows]


def _raf_class_f1_at_best(payload: dict[str, Any], best_epoch: int | None) -> dict[str, float]:
    history = _history(payload)
    extra = history.get("extra_val") or []
    if best_epoch and 1 <= best_epoch <= len(extra):
        row = extra[best_epoch - 1]
        raf = row.get("raf_db") if isinstance(row, dict) else None
        if isinstance(raf, dict) and isinstance(raf.get("class_f1"), dict):
            return {k: float(v) for k, v in raf["class_f1"].items() if _as_float(v) is not None}
    metrics = _metrics(payload)
    if isinstance(metrics.get("class_f1"), dict):
        return {
            k: float(v) for k, v in metrics["class_f1"].items() if _as_float(v) is not None
        }
    per_class = metrics.get("per_class")
    if isinstance(per_class, dict):
        out: dict[str, float] = {}
        for name, stats in per_class.items():
            if isinstance(stats, dict) and _as_float(stats.get("f1")) is not None:
                out[name] = float(stats["f1"])
        return out
    extra_final = metrics.get("final_extra_val") or metrics.get("extra_val")
    if isinstance(extra_final, dict) and isinstance(extra_final.get("raf_db"), dict):
        raf = extra_final["raf_db"]
        if isinstance(raf.get("class_f1"), dict):
            return {k: float(v) for k, v in raf["class_f1"].items() if _as_float(v) is not None}
    return {}


def _plateau(
    fer_acc: list[float | None],
    monitor: list[float | None],
) -> tuple[bool | None, float | None, bool | None]:
    """Return (plateau, last3_fer_span_pp, climbing)."""
    if len(fer_acc) < 3 or len(monitor) < 2:
        climbing = None
        if len(monitor) >= 2:
            a, b = _as_float(monitor[-2]), _as_float(monitor[-1])
            climbing = a is not None and b is not None and b >= a
        return None, _span_pp(fer_acc[-3:]) if fer_acc else None, climbing

    span = _span_pp(fer_acc[-3:])
    peak, peak_epoch = _best_with_epoch(monitor)
    last = _as_float(monitor[-1])
    bounced = (
        peak is not None
        and last is not None
        and peak_epoch is not None
        and peak_epoch < len(monitor)
        and last < peak
    )
    plateau = span is not None and span <= FER_PLATEAU_SPAN_PP and bounced
    prev, cur = _as_float(monitor[-2]), last
    climbing = prev is not None and cur is not None and cur >= prev
    return plateau, span, climbing


def _abort_classes(history: dict[str, Any]) -> list[str]:
    aborted: list[str] = []
    for name in TARGET_CLASSES:
        f1_last = _as_float(_last(_class_series(history, "val_class_f1", name)))
        ce_last = _as_float(_last(_class_series(history, "val_class_loss", name)))
        if (
            f1_last is not None
            and ce_last is not None
            and ce_last >= ABORT_CE
            and ABORT_F1_LO <= f1_last <= ABORT_F1_HI
        ):
            aborted.append(name)
    return aborted


def summarize_run(payload: dict[str, Any], source: str | Path | None = None) -> dict[str, Any]:
    """Flatten a train report or evaluate report. Never includes ``history``."""
    hp = _hp(payload)
    training = hp.get("training") if isinstance(hp.get("training"), dict) else {}
    scheduler = hp.get("scheduler") if isinstance(hp.get("scheduler"), dict) else {}
    metrics = _metrics(payload)
    history = _history(payload)

    monitor = str(training.get("monitor_metric") or payload.get("best_metric_name") or "macro_f1")
    fer_acc = [_as_float(v) for v in (history.get("val_accuracy") or [])]
    fer_f1 = [_as_float(v) for v in (history.get("val_macro_f1") or [])]
    monitor_series = _monitor_series(payload, monitor)
    raf_f1 = _extra_series(history, "raf_db", "macro_f1")

    n_epochs = len(fer_acc) or len(monitor_series) or len(fer_f1)
    best_fer, best_fer_ep = _best_with_epoch(fer_acc)
    best_mon, best_mon_ep = _best_with_epoch(monitor_series)
    raf_best, _ = _best_with_epoch(raf_f1)
    plateau, span, climbing = _plateau(fer_acc, monitor_series or fer_f1)

    final_acc = _as_float(_last(fer_acc))
    if final_acc is None:
        final_acc = _as_float(metrics.get("final_val_accuracy") or metrics.get("accuracy"))
    final_f1 = _as_float(_last(fer_f1))
    if final_f1 is None:
        final_f1 = _as_float(metrics.get("final_val_macro_f1") or metrics.get("macro_f1"))

    stamp = payload.get("timestamp_utc") or metrics.get("run_stamp")
    if stamp is None and source is not None:
        name = Path(source).stem
        stamp = name[4:] if name.startswith("run_") else name

    summary: dict[str, Any] = {
        "stamp": stamp,
        "source": str(source) if source is not None else None,
        "lr": training.get("lr"),
        "epochs": training.get("epochs"),
        "class_weight_overrides": training.get("class_weight_overrides") or {},
        "class_weight_multipliers": training.get("class_weight_multipliers") or {},
        "label_smoothing": training.get("label_smoothing"),
        "monitor_metric": monitor,
        "resume_from_best": training.get("resume_from_best"),
        "eta_min_ratio": scheduler.get("eta_min_ratio"),
        "prior_best_metric": metrics.get("prior_best_metric"),
        "resumed_from": metrics.get("resumed_from"),
        "data_version": metrics.get("data_version"),
        "final_val_accuracy": final_acc,
        "final_val_macro_f1": final_f1,
        "best_fer_acc": best_fer if best_fer is not None else final_acc,
        "best_fer_acc_epoch": best_fer_ep,
        "best_monitor": best_mon if best_mon is not None else _as_float(payload.get("best_metric_value")),
        "best_monitor_epoch": best_mon_ep,
        "raf_db_macro_f1_best": raf_best,
        "raf_db_macro_f1_last": _as_float(_last(raf_f1)),
        "n_epochs": n_epochs,
        "last3_fer_acc_span_pp": span,
        "plateau": plateau,
        "climbing": climbing,
        "abort_classes": _abort_classes(history),
        "raf_class_f1_at_best": _raf_class_f1_at_best(payload, best_mon_ep),
        "fer_acc_at_best_monitor": (
            _as_float(fer_acc[best_mon_ep - 1])
            if best_mon_ep and 1 <= best_mon_ep <= len(fer_acc)
            else final_acc
        ),
    }
    for name in TARGET_CLASSES:
        f1s = _class_series(history, "val_class_f1", name)
        ces = _class_series(history, "val_class_loss", name)
        f1_best, _ = _best_with_epoch(f1s)
        ce_max, _ = _best_with_epoch(ces)
        summary[f"{name}_f1_last"] = _as_float(_last(f1s))
        summary[f"{name}_f1_best"] = f1_best
        summary[f"{name}_ce_last"] = _as_float(_last(ces))
        summary[f"{name}_ce_max"] = ce_max
        if summary[f"{name}_f1_last"] is None:
            final_map = metrics.get("final_val_class_f1") or {}
            if isinstance(final_map, dict):
                summary[f"{name}_f1_last"] = _as_float(final_map.get(name))
                if summary[f"{name}_f1_best"] is None:
                    summary[f"{name}_f1_best"] = summary[f"{name}_f1_last"]
        if summary[f"{name}_ce_last"] is None:
            final_ce = metrics.get("final_val_class_loss") or {}
            if isinstance(final_ce, dict):
                summary[f"{name}_ce_last"] = _as_float(final_ce.get(name))
                if summary[f"{name}_ce_max"] is None:
                    summary[f"{name}_ce_max"] = summary[f"{name}_ce_last"]

    missing = REQUIRED_SUMMARY_KEYS - summary.keys()
    if missing:
        raise KeyError(f"summarize_run missing keys: {sorted(missing)}")
    return summary


def load_run(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{p} is not a JSON object")
    return summarize_run(payload, source=p)


def list_run_paths(log_dir: str | Path) -> list[Path]:
    d = Path(log_dir)
    if not d.is_absolute():
        d = project_root() / d
    if not d.is_dir():
        return []
    return sorted(d.glob("run_*.json"))


def all_time_highs(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    raf_best: float | None = None
    raf_stamp = None
    fer_best: float | None = None
    fer_stamp = None
    for row in summaries:
        raf = _as_float(row.get("raf_db_macro_f1_best"))
        if raf is not None and (raf_best is None or raf > raf_best):
            raf_best, raf_stamp = raf, row.get("stamp")
        fer = _as_float(row.get("best_fer_acc"))
        if fer is not None and (fer_best is None or fer > fer_best):
            fer_best, fer_stamp = fer, row.get("stamp")
    return {
        "raf_db_macro_f1": {"value": raf_best, "stamp": raf_stamp},
        "fer_val_accuracy": {"value": fer_best, "stamp": fer_stamp},
    }


def _pp_delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return (new - old) * 100.0


def compare_pair(previous: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Deltas (percentage points for rates) plus progress / plateau / abort."""
    raf_pp = _pp_delta(candidate.get("raf_db_macro_f1_best"), previous.get("raf_db_macro_f1_best"))
    fer_pp = _pp_delta(candidate.get("best_fer_acc"), previous.get("best_fer_acc"))
    class_deltas: dict[str, Any] = {}
    class_progress: list[str] = []
    for name in TARGET_CLASSES:
        f1_pp = _pp_delta(candidate.get(f"{name}_f1_best"), previous.get(f"{name}_f1_best"))
        ce_delta = None
        new_ce = _as_float(candidate.get(f"{name}_ce_last"))
        old_ce = _as_float(previous.get(f"{name}_ce_last"))
        if new_ce is not None and old_ce is not None:
            ce_delta = new_ce - old_ce
        exploded = new_ce is not None and new_ce >= ABORT_CE
        rose = f1_pp is not None and f1_pp >= CLASS_F1_PROGRESS * 100.0
        if rose and not exploded:
            class_progress.append(name)
        class_deltas[name] = {"f1_pp": f1_pp, "ce_delta": ce_delta, "ce_exploded": exploded}

    hp_keys = (
        "lr",
        "epochs",
        "class_weight_overrides",
        "class_weight_multipliers",
        "label_smoothing",
        "monitor_metric",
        "resume_from_best",
        "eta_min_ratio",
        "data_version",
    )
    hp_changed = {
        k: {"from": previous.get(k), "to": candidate.get(k)}
        for k in hp_keys
        if previous.get(k) != candidate.get(k)
    }

    new_high_raf = raf_pp is not None and raf_pp > 0
    new_high_fer = fer_pp is not None and fer_pp > 0
    progress = bool(new_high_raf or new_high_fer or class_progress)
    return {
        "previous_stamp": previous.get("stamp"),
        "candidate_stamp": candidate.get("stamp"),
        "raf_db_macro_f1_pp": raf_pp,
        "fer_val_accuracy_pp": fer_pp,
        "classes": class_deltas,
        "hp_changed": hp_changed,
        "progress": progress,
        "new_high_raf": new_high_raf,
        "new_high_fer": new_high_fer,
        "class_progress": class_progress,
        "plateau": candidate.get("plateau"),
        "abort_classes": list(candidate.get("abort_classes") or []),
        "climbing": candidate.get("climbing"),
    }


def pareto_for(summary: dict[str, Any]) -> dict[str, Any] | None:
    class_f1 = summary.get("raf_class_f1_at_best")
    if not isinstance(class_f1, dict) or not class_f1:
        return None
    payload = {
        "class_f1": class_f1,
        "metrics": {"accuracy": summary.get("fer_acc_at_best_monitor")},
    }
    result = evaluate_payload(payload, fer_acc=_as_float(summary.get("fer_acc_at_best_monitor")))
    return result.as_dict()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def checkpoint_match(stamp: str | None, checkpoint_dir: str | Path | None) -> dict[str, Any] | None:
    if not stamp or checkpoint_dir is None:
        return None
    d = Path(checkpoint_dir)
    if not d.is_absolute():
        d = project_root() / d
    best = d / "best.pt"
    run_pt = d / f"run_{stamp}.pt"
    if not best.is_file() or not run_pt.is_file():
        return {"best_exists": best.is_file(), "run_pt_exists": run_pt.is_file(), "match": None}
    return {
        "best_exists": True,
        "run_pt_exists": True,
        "match": sha256_file(best) == sha256_file(run_pt),
        "best": str(best),
        "run_pt": str(run_pt),
    }


def build_report(
    paths: list[Path],
    *,
    against: list[Path] | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    runs = [load_run(p) for p in paths]
    pool = list(runs)
    if against:
        pool.extend(load_run(p) for p in against)
    for row in runs:
        if checkpoint_dir is not None:
            row["checkpoint"] = checkpoint_match(row.get("stamp"), checkpoint_dir)
        try:
            row["pareto"] = pareto_for(row)
        except (ValueError, KeyError):
            row["pareto"] = None

    compare = None
    if against and runs:
        prev = load_run(against[-1]) if against else None
        if prev is not None:
            compare = compare_pair(prev, runs[-1])
    elif len(runs) >= 2:
        compare = compare_pair(runs[-2], runs[-1])

    return {
        "runs": runs,
        "all_time": all_time_highs(pool),
        "compare": compare,
    }


def format_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    all_time = report.get("all_time") or {}
    raf = (all_time.get("raf_db_macro_f1") or {}).get("value")
    fer = (all_time.get("fer_val_accuracy") or {}).get("value")
    lines.append(
        "all-time  raf_db.macro_f1="
        + (f"{raf:.4f}" if raf is not None else "n/a")
        + f" ({(all_time.get('raf_db_macro_f1') or {}).get('stamp')})"
        + "  fer_acc="
        + (f"{fer:.4f}" if fer is not None else "n/a")
        + f" ({(all_time.get('fer_val_accuracy') or {}).get('stamp')})"
    )
    header = (
        f"{'stamp':<20} {'raf_best':>8} {'fer_best':>8} {'fear_f1':>8} "
        f"{'fear_ce':>8} {'sad_f1':>8} {'plat':>5} {'abort':<8} climb"
    )
    lines.append(header)
    for row in report.get("runs") or []:
        abort = ",".join(row.get("abort_classes") or []) or "-"
        plat = row.get("plateau")
        plat_s = "-" if plat is None else ("Y" if plat else "n")
        climb = row.get("climbing")
        climb_s = "-" if climb is None else ("Y" if climb else "n")
        raf_v = row.get("raf_db_macro_f1_best")
        fer_v = row.get("best_fer_acc")
        lines.append(
            f"{str(row.get('stamp') or '-'):<20} "
            f"{(f'{raf_v:.4f}' if raf_v is not None else 'n/a'):>8} "
            f"{(f'{fer_v:.4f}' if fer_v is not None else 'n/a'):>8} "
            f"{(f'{row.get('fear_f1_last'):.3f}' if row.get('fear_f1_last') is not None else 'n/a'):>8} "
            f"{(f'{row.get('fear_ce_last'):.3f}' if row.get('fear_ce_last') is not None else 'n/a'):>8} "
            f"{(f'{row.get('sad_f1_last'):.3f}' if row.get('sad_f1_last') is not None else 'n/a'):>8} "
            f"{plat_s:>5} {abort:<8} {climb_s}"
        )
    cmp_ = report.get("compare")
    if cmp_:
        lines.append(
            "compare "
            f"{cmp_.get('previous_stamp')} → {cmp_.get('candidate_stamp')} | "
            f"progress={cmp_.get('progress')} plateau={cmp_.get('plateau')} "
            f"abort={cmp_.get('abort_classes')} "
            f"raf_pp={cmp_.get('raf_db_macro_f1_pp')} fer_pp={cmp_.get('fer_val_accuracy_pp')} "
            f"hp_changed={list((cmp_.get('hp_changed') or {}).keys())}"
        )
    return "\n".join(lines) + "\n"


def _resolve_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    paths = [Path(p) for p in (args.runs or [])]
    if args.dir:
        paths.extend(list_run_paths(args.dir))
    against = list_run_paths(args.against) if args.against else []
    if not paths:
        for rel in DEFAULT_LOG_DIRS:
            found = list_run_paths(rel)
            if found:
                paths = found
                break
    if not paths:
        raise SystemExit(
            "No run_*.json found. Pass files, --dir, or put logs in "
            "artifacts/models/efficientnet_b0/logs/"
        )
    return paths, against


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten and compare FER run_*.json reports (no nested history walk)."
    )
    parser.add_argument("runs", nargs="*", help="run_*.json or metrics.json paths")
    parser.add_argument(
        "--dir",
        help="Directory of run_*.json (e.g. artifacts/models/efficientnet_b0/logs)",
    )
    parser.add_argument(
        "--against",
        help="Older log dir for all-time + previous snapshot (Colab canonical store)",
    )
    parser.add_argument(
        "--checkpoints",
        help="Directory with best.pt and run_<stamp>.pt for hash compare",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="json is the agent-facing compact payload; text is a one-screen table",
    )
    args = parser.parse_args()
    paths, against = _resolve_paths(args)
    report = build_report(paths, against=against, checkpoint_dir=args.checkpoints)
    if args.format == "text":
        sys.stdout.write(format_text(report))
        return
    json.dump(report, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
