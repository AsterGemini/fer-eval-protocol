#!/usr/bin/env bash
# A/B experiment harness: train baseline vs candidate configs fresh-seed,
# evaluate both, append one row each to artifacts/experiments.csv.
#
# Usage (from fer-system/):
#   bash scripts/ab_run.sh configs/public.yaml path/to/candidate.yaml
#
# Requires: python env with project deps activated.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASELINE_CFG="${1:-configs/public.yaml}"
CANDIDATE_CFG="${2:?Usage: ab_run.sh <baseline_config> <candidate_config>}"
CSV="${3:-artifacts/experiments.csv}"

mkdir -p "$(dirname "$CSV")"

if [[ ! -f "$CSV" ]]; then
  echo "timestamp_utc,config,run_stamp,macro_f1,accuracy,val_loss,best_checkpoint,notes" > "$CSV"
fi

run_one() {
  local label="$1"
  local config="$2"
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local out_dir="artifacts/ab/${label}_${stamp}"
  mkdir -p "$out_dir"

  echo "=== A/B | ${label} | config=${config} | stamp=${stamp} ==="

  # Isolate best.pt so A and B do not clobber each other mid-run.
  local isolated_best="${out_dir}/best.pt"
  python - <<PY
from pathlib import Path
import yaml

cfg_path = Path("${config}")
with cfg_path.open() as f:
    cfg = yaml.safe_load(f)
cfg["training"]["resume_from_best"] = False
cfg["paths"]["best_checkpoint"] = "${isolated_best}"
cfg["paths"]["figure_dir"] = "${out_dir}"
cfg["paths"]["log_dir"] = "${out_dir}"
cfg["paths"]["checkpoint_dir"] = "${out_dir}"
out = Path("${out_dir}/resolved_config.yaml")
with out.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(out)
PY

  python -m src.train --config "${out_dir}/resolved_config.yaml"
  python -m src.evaluate \
    --config "${out_dir}/resolved_config.yaml" \
    --checkpoint "${isolated_best}" \
    --output-dir "${out_dir}"

  python - <<PY
import json
from pathlib import Path

metrics_path = Path("${out_dir}/metrics.json")
data = json.loads(metrics_path.read_text())
# evaluate writes {"metrics": {...}} ; train report is different shape
m = data.get("metrics", data)
macro = m.get("macro_f1", m.get("best_macro_f1"))
acc = m.get("accuracy", m.get("final_val_accuracy"))
loss = m.get("loss", m.get("final_val_loss"))
row = ",".join([
    "${stamp}",
    "${config}",
    "${stamp}",
    f"{macro:.6f}" if macro is not None else "",
    f"{acc:.6f}" if acc is not None else "",
    f"{loss:.6f}" if loss is not None else "",
    "${isolated_best}",
    "${label}",
])
with open("${CSV}", "a") as f:
    f.write(row + "\n")
print("Appended:", row)
PY
}

run_one "baseline" "$BASELINE_CFG"
run_one "candidate" "$CANDIDATE_CFG"

echo "A/B complete. Results → ${CSV}"
