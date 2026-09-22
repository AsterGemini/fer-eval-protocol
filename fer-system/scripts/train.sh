#!/usr/bin/env bash
# Launch selectable FER training from the fer-system/ root.
# Examples:
#   bash scripts/train.sh --model b0
#   bash scripts/train.sh --model b1
#   bash scripts/train.sh --model b2
#   bash scripts/train.sh --model b2 --epochs 200 --patience 16 --full-backbone
#   bash scripts/train.sh --model b1 --dry-run
# An optional config path may still be the first positional argument.
# Also writes per-run copies as run_<stamp>.pt when macro F1 improves.
#
# One command on Colab too: kagglehub download, ExpW extract/crop, and
# p7zip install happen inside src.train / prepare_expw. Do not run
# data.download first unless you want to refresh data/manifest.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/public.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG="$1"
  shift
fi

# Leftover local/Drive ImageFolder crops are unused; prepared ExpW lives in
# ~/.cache/fer-system/expw. Ignore if the path is absent.
rm -rf "$ROOT/data/prepared/expw"
if [ -d /content/drive/MyDrive/Facial-Emo-reco/fer-system/data/prepared/expw ]; then
  rm -rf /content/drive/MyDrive/Facial-Emo-reco/fer-system/data/prepared/expw
fi

echo "Working directory: $ROOT"
echo "Config: $CONFIG"
python -m src.train --config "$CONFIG" "$@"
