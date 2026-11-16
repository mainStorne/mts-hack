#!/usr/bin/env bash
# Сквозной прогон на CPU: сырой jsonl звонков → очистка ASR-шума → извлечение фактов.
#   ./predict.sh data/val.jsonl pred_val.jsonl
set -euo pipefail
cd "$(dirname "$0")"
IN="$1"; OUT="$2"; TMP="$(mktemp --suffix=.jsonl)"
python3 asr_denoise.py fix data/asr_model_train.pkl "$IN" -o "$TMP"
python3 pipeline.py predict model_full.pkl "$TMP" -o "$OUT"
rm -f "$TMP"
