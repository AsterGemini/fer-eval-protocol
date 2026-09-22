"""Sample FER val errors so a human can audit official labels.

Writes a self-contained review pack:

    data/label_review/val_errors/
      images/NN.jpg
      review.json
      index.html

Open ``index.html`` in a browser, mark each official label Correct / Wrong /
Unsure, then download ``judgments.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from data.download import get_dataset_paths
from data.transforms import build_transforms
from src.dataset import FERDataset
from src.inference import load_predictor
from src.utils import (
    ensure_dir,
    load_config,
    project_root,
    set_seed,
    setup_logging,
)

logger = logging.getLogger("fer.sample_val_errors")

DEFAULT_N = 25
MINORITY_CLASSES = ("disgust", "fear")
MINORITY_QUOTA = 8
DEFAULT_OUT = "data/label_review/val_errors"


def collect_errors(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep rows where the predicted label disagrees with the official label."""
    return [row for row in records if row["official_label"] != row["predicted_label"]]


def sample_errors(
    errors: list[dict[str, Any]],
    n: int,
    *,
    seed: int,
    minority_classes: tuple[str, ...] = MINORITY_CLASSES,
    minority_quota: int = MINORITY_QUOTA,
) -> list[dict[str, Any]]:
    """Stratified sample: fill minority quotas first, then other classes.

    Within a class, higher model confidence is preferred (better label-noise probe).
    """
    if n <= 0:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in errors:
        grouped[str(row["official_label"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: (-float(r["confidence"]), str(r["path"])))

    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    used_paths: set[str] = set()

    def take(label: str, k: int) -> None:
        pool = [r for r in grouped.get(label, []) if r["path"] not in used_paths]
        k = min(k, len(pool))
        for row in pool[:k]:
            picked.append(row)
            used_paths.add(row["path"])

    for label in minority_classes:
        take(label, minority_quota)

    remaining = n - len(picked)
    other_labels = sorted(name for name in grouped if name not in minority_classes)
    rng.shuffle(other_labels)
    if remaining > 0 and other_labels:
        per = max(1, remaining // len(other_labels))
        for label in other_labels:
            if len(picked) >= n:
                break
            take(label, min(per, n - len(picked)))

    if len(picked) < n:
        leftovers = [r for r in errors if r["path"] not in used_paths]
        leftovers.sort(key=lambda r: (-float(r["confidence"]), str(r["path"])))
        for row in leftovers:
            if len(picked) >= n:
                break
            picked.append(row)
            used_paths.add(row["path"])

    picked.sort(key=lambda r: (str(r["official_label"]), -float(r["confidence"]), str(r["path"])))
    return picked[:n]


def _write_index_html(out_dir: Path, items: list[dict[str, Any]], class_names: list[str]) -> None:
    cards = []
    for item in items:
        probs = item.get("all_probabilities") or {}
        prob_rows = "".join(
            f"<tr><td>{escape(name)}</td><td>{100 * float(probs.get(name, 0)):.1f}%</td></tr>"
            for name in class_names
        )
        cards.append(
            f"""
<article class="card" data-id="{escape(item['id'])}">
  <img src="{escape(item['rel_image'])}" alt="val error {escape(item['id'])}" />
  <h2>#{escape(item['id'])} official <em>{escape(item['official_label'])}</em></h2>
  <p>Model: <strong>{escape(item['predicted_label'])}</strong>
     ({100 * float(item['confidence']):.1f}%)</p>
  <table>{prob_rows}</table>
  <p class="q">Is the <em>official</em> label correct?</p>
  <div class="btns">
    <button type="button" data-verdict="correct">Correct</button>
    <button type="button" data-verdict="wrong">Wrong</button>
    <button type="button" data-verdict="unsure">Unsure</button>
  </div>
  <p class="status">Not judged yet</p>
</article>
"""
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>FER val-error label review</title>
  <style>
    body {{ font-family: sans-serif; margin: 1.5rem; background: #111; color: #eee; }}
    h1 {{ font-size: 1.2rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; }}
    .card {{ background: #1c1c1c; padding: 0.75rem; border-radius: 8px; }}
    img {{ width: 100%; height: auto; image-rendering: pixelated; background: #000; }}
    table {{ width: 100%; font-size: 0.8rem; }}
    button {{ margin-right: 0.3rem; }}
    button.active {{ outline: 2px solid #6cf; }}
    .done {{ margin: 1rem 0; }}
  </style>
</head>
<body>
  <h1>25 FER val errors — judge the official label, not the model</h1>
  <p>For each face: if you agree with the official emotion, mark <strong>Correct</strong>.
     If the official label is wrong (or not a face), mark <strong>Wrong</strong>.</p>
  <p class="done">Judged: <span id="count">0</span> / {len(items)}
     <button type="button" id="download">Download judgments.json</button></p>
  <div class="grid">
    {''.join(cards)}
  </div>
  <script>
    const items = {json.dumps({item['id']: None for item in items})};
    const storeKey = "fer_val_error_judgments";
    const saved = JSON.parse(localStorage.getItem(storeKey) || "{{}}");
    Object.assign(items, saved);
    function refresh() {{
      let n = 0;
      document.querySelectorAll(".card").forEach(card => {{
        const id = card.dataset.id;
        const verdict = items[id];
        card.querySelectorAll("button[data-verdict]").forEach(b => {{
          b.classList.toggle("active", b.dataset.verdict === verdict);
        }});
        card.querySelector(".status").textContent = verdict ? ("You: " + verdict) : "Not judged yet";
        if (verdict) n += 1;
      }});
      document.getElementById("count").textContent = String(n);
      localStorage.setItem(storeKey, JSON.stringify(items));
    }}
    document.querySelectorAll(".card button[data-verdict]").forEach(btn => {{
      btn.addEventListener("click", () => {{
        items[btn.closest(".card").dataset.id] = btn.dataset.verdict;
        refresh();
      }});
    }});
    document.getElementById("download").addEventListener("click", () => {{
      const blob = new Blob([JSON.stringify(items, null, 2)], {{type: "application/json"}});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "judgments.json";
      a.click();
    }});
    refresh();
  </script>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def write_review_pack(
    sampled: list[dict[str, Any]],
    out_dir: Path,
    class_names: list[str],
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy sampled faces into ``out_dir`` and write JSON + HTML."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    for i, row in enumerate(sampled, start=1):
        src = Path(row["path"])
        dest_name = f"{i:02d}_{row['official_label']}_pred-{row['predicted_label']}.png"
        dest = images_dir / dest_name
        with Image.open(src) as im:
            im.convert("RGB").save(dest, format="PNG")
        item = {
            "id": f"{i:02d}",
            "official_label": row["official_label"],
            "predicted_label": row["predicted_label"],
            "confidence": float(row["confidence"]),
            "official_probability": float(row["official_probability"]),
            "all_probabilities": row["all_probabilities"],
            "source_path": str(src),
            "rel_image": f"images/{dest_name}",
            "human_verdict": None,
        }
        items.append(item)

    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": (
            "For each image, decide whether the official_label is correct. "
            "Ignore whether the model is right."
        ),
        "class_names": list(class_names),
        "n": len(items),
        "items": items,
        "meta": extra_meta or {},
    }
    (out_dir / "review.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_index_html(out_dir, items, class_names)
    return payload


@torch.no_grad()
def predict_val_records(
    cfg: dict[str, Any],
    checkpoint: str | Path,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """Run the checkpoint on the primary val split and return per-image records."""
    class_names = list(cfg["class_names"])
    _, val_dir = get_dataset_paths(cfg)
    ds = FERDataset(
        val_dir,
        class_names,
        transform=build_transforms(cfg, train=False),
        source_name="primary_val",
    )
    model, device, _ = load_predictor(cfg, checkpoint)
    model.eval()

    bs = int(batch_size or cfg["training"]["batch_size"])
    records: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(ds), bs), desc="val-errors"):
        end = min(start + bs, len(ds))
        tensors = []
        labels = []
        paths = []
        for i in range(start, end):
            image, label = ds[i]
            tensors.append(image)
            labels.append(label)
            paths.append(ds.samples[i][0])
        batch = torch.stack(tensors, dim=0).to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu()
        pred_idx = probs.argmax(dim=1).tolist()
        for i, path in enumerate(paths):
            official_i = int(labels[i])
            pred_i = int(pred_idx[i])
            row_probs = {name: float(probs[i, j]) for j, name in enumerate(class_names)}
            records.append(
                {
                    "path": str(path),
                    "official_label": class_names[official_i],
                    "predicted_label": class_names[pred_i],
                    "confidence": float(probs[i, pred_i]),
                    "official_probability": float(probs[i, official_i]),
                    "all_probabilities": row_probs,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample FER val errors for label review")
    parser.add_argument("--config", type=str, default="configs/public.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging("fer.sample_val_errors", log_dir=cfg["paths"]["log_dir"])
    set_seed(int(cfg["seed"]))
    ckpt = args.checkpoint or cfg["paths"]["best_checkpoint"]
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root() / out_dir
    ensure_dir(out_dir.parent)

    records = predict_val_records(cfg, ckpt)
    errors = collect_errors(records)
    sampled = sample_errors(errors, int(args.n), seed=int(cfg["seed"]))
    if len(sampled) < int(args.n):
        logger.warning("Only %d val errors available; requested %d", len(sampled), args.n)

    by_official: dict[str, int] = defaultdict(int)
    for row in sampled:
        by_official[row["official_label"]] += 1
    payload = write_review_pack(
        sampled,
        out_dir,
        list(cfg["class_names"]),
        extra_meta={
            "checkpoint": str(ckpt),
            "val_size": len(records),
            "n_errors": len(errors),
            "error_rate": (len(errors) / len(records)) if records else None,
            "sampled_by_official_label": dict(by_official),
        },
    )
    logger.info(
        "Wrote %d val-error cards → %s (from %d errors / %d val)",
        payload["n"],
        out_dir / "index.html",
        len(errors),
        len(records),
    )
    logger.info("Sample mix by official label: %s", dict(by_official))


if __name__ == "__main__":
    main()
