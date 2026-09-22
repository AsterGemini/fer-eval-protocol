"""Apply the 25-error human audit to primary FER val labels.

DISABLED (2026-08-24): ``data.val_label_corrections`` is null in
configs/public.yaml and must stay null. Remapping "wrong" official labels to
the audit-time model prediction is circular evaluation — the human verdicts
never named a replacement class, and the model was confidently wrong on 8 of
21 decided audit cases. This module remains for audit tooling and tests only;
the JSON it writes is documentation, not an eval input.

Unsure rows are dropped. Wrong official labels are remapped to the model
prediction at audit time. Correct official labels are left alone. The
kagglehub ImageFolder is not rewritten.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.utils import project_root

logger = logging.getLogger("fer.val_corrections")

DEFAULT_PATH = "data/label_review/val_corrections.json"


def sample_key(path: Path) -> str:
    """``<official_folder>/<filename>`` — stable across kagglehub layouts."""
    path = Path(path)
    return f"{path.parent.name}/{path.name}"


def relpath_from_source(source_path: str | Path) -> str:
    return sample_key(Path(source_path))


def build_corrections(
    review: dict[str, Any],
    judgments: dict[str, str],
) -> dict[str, Any]:
    """Turn review items + human verdicts into drop/remap maps."""
    items_out: list[dict[str, Any]] = []
    drop: list[str] = []
    remap: dict[str, str] = {}

    for item in review.get("items") or []:
        item_id = str(item["id"])
        if item_id not in judgments:
            raise KeyError(f"Missing judgment for id {item_id}")
        verdict = str(judgments[item_id]).strip().lower()
        if verdict not in {"correct", "wrong", "unsure"}:
            raise ValueError(f"Unknown verdict {verdict!r} for id {item_id}")
        relpath = relpath_from_source(item["source_path"])
        official = str(item["official_label"])
        predicted = str(item["predicted_label"])
        corrected = official
        action = "keep"
        if verdict == "unsure":
            action = "drop"
            drop.append(relpath)
            corrected = None
        elif verdict == "wrong":
            action = "remap"
            remap[relpath] = predicted
            corrected = predicted
        items_out.append(
            {
                "id": item_id,
                "verdict": verdict,
                "action": action,
                "relpath": relpath,
                "official_label": official,
                "corrected_label": corrected,
            }
        )

    drop_sorted = sorted(set(drop))
    return {
        "version": 1,
        "policy": (
            "Drop unsure. Remap wrong official labels to the audit-time "
            "model prediction. Keep correct official labels. Overlay only; "
            "do not rewrite the kagglehub val tree."
        ),
        "n_items": len(items_out),
        "n_drop": len(drop_sorted),
        "n_remap": len(remap),
        "n_keep": sum(1 for row in items_out if row["action"] == "keep"),
        "drop": drop_sorted,
        "remap": dict(sorted(remap.items())),
        "items": items_out,
    }


def load_corrections(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or str(path).strip() in {"", "null", "None"}:
        return None
    loc = Path(path)
    if not loc.is_absolute():
        loc = project_root() / loc
    if not loc.is_file():
        raise FileNotFoundError(f"val_label_corrections not found: {loc}")
    with loc.open(encoding="utf-8") as f:
        spec = json.load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"corrections file must be an object: {loc}")
    return spec


def load_corrections_from_cfg(cfg: dict[str, Any]) -> dict[str, Any] | None:
    return load_corrections(cfg.get("data", {}).get("val_label_corrections"))


def apply_corrections(dataset: Any, spec: dict[str, Any]) -> dict[str, int]:
    """Filter/remap ``dataset.samples`` in place. ``dataset`` is a FERDataset."""
    drop = set(spec.get("drop") or [])
    remap = dict(spec.get("remap") or {})
    class_to_idx = dataset.class_to_idx
    for name in remap.values():
        if name not in class_to_idx:
            raise KeyError(f"Unknown remap class {name!r}")

    kept: list[tuple[Path, int]] = []
    n_drop = 0
    n_remap = 0
    seen_drop: set[str] = set()
    seen_remap: set[str] = set()
    for path, label in dataset.samples:
        key = sample_key(path)
        if key in drop:
            n_drop += 1
            seen_drop.add(key)
            continue
        new_name = remap.get(key)
        if new_name is not None:
            label = class_to_idx[new_name]
            n_remap += 1
            seen_remap.add(key)
        kept.append((path, label))

    missing_drop = sorted(drop - seen_drop)
    missing_remap = sorted(set(remap) - seen_remap)
    expected = len(drop) + len(remap)
    if expected and (n_drop + n_remap) == 0:
        raise RuntimeError(
            "val_label_corrections matched 0 val samples. "
            f"missing_drop={missing_drop} missing_remap={missing_remap}"
        )
    if missing_drop or missing_remap:
        logger.warning(
            "val_label_corrections unmatched drop=%s remap=%s",
            missing_drop,
            missing_remap,
        )

    dataset.samples = kept
    stats = {
        "dropped": n_drop,
        "remapped": n_remap,
        "kept": len(kept),
    }
    logger.info(
        "Val label corrections: dropped %d, remapped %d, val size %d",
        stats["dropped"],
        stats["remapped"],
        stats["kept"],
    )
    return stats


def apply_corrections_from_cfg(dataset: Any, cfg: dict[str, Any]) -> dict[str, int] | None:
    spec = load_corrections_from_cfg(cfg)
    if spec is None:
        return None
    return apply_corrections(dataset, spec)


def write_corrections(
    review_path: str | Path,
    judgments_path: str | Path,
    out_path: str | Path,
) -> dict[str, Any]:
    review = json.loads(Path(review_path).read_text(encoding="utf-8"))
    judgments = json.loads(Path(judgments_path).read_text(encoding="utf-8"))
    spec = build_corrections(review, judgments)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec


def persist_verdicts(review_path: str | Path, judgments: dict[str, str]) -> None:
    """Write human_verdict onto review.json items (eval metadata only)."""
    path = Path(review_path)
    review = json.loads(path.read_text(encoding="utf-8"))
    for item in review.get("items") or []:
        item_id = str(item["id"])
        if item_id in judgments:
            item["human_verdict"] = judgments[item_id]
            if judgments[item_id] == "wrong":
                item["corrected_label"] = item["predicted_label"]
            elif judgments[item_id] == "correct":
                item["corrected_label"] = item["official_label"]
            else:
                item["corrected_label"] = None
    path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = project_root()
    spec = write_corrections(
        root / "data/label_review/val_errors/review.json",
        root / "judgments.json",
        root / DEFAULT_PATH,
    )
    persist_verdicts(root / "data/label_review/val_errors/review.json", json.loads(
        (root / "judgments.json").read_text(encoding="utf-8")
    ))
    print(
        f"Wrote {DEFAULT_PATH}: drop={spec['n_drop']} "
        f"remap={spec['n_remap']} keep={spec['n_keep']}"
    )


if __name__ == "__main__":
    main()
