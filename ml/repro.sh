#!/usr/bin/env bash
# Воспроизведение конвейера CFE. Запуск из ml/. Подкоманды: verify | fetch | data | gliner | gemma | ensemble | all
# Переменные: GEMMA_BASE (каталог gemma-3-1b-it; обязательна для data/gemma), SMOKE=1 (короткий прогон -> runs/repro-smoke, adapters/repro-smoke),
#             SNAPSHOT (путь к cfe-inputs-v1.tar.zst для `fetch`), OUT (каталог адаптера GLiNER, по умолчанию runs/struct-lora-repro).
set -euo pipefail
cd "$(dirname "$0")"
SMOKE="${SMOKE:-0}"
NAME="gemma-v21-targeted-gen-repro"
OUT="${OUT:-runs/struct-lora-repro}"
GL_ARGS="--epochs 3 --batch-size 8 --accum 2 --data struct"   # repro/recorded/struct-lora/training_config.json
GM_DIR=gemma/data/targeted_gen_v21_ctx3k
PY="${PY:-python}"
if [ "$SMOKE" = 1 ]; then
  OUT=runs/repro-smoke/struct-lora; NAME=_smoke   # train.py --smoke всегда пишет в adapters/_smoke; ниже переименовывается в adapters/repro-smoke
  GL_ARGS="--epochs 1 --batch-size 8 --accum 2 --data struct --max-train 64"
fi

link_raw() { mkdir -p gemma/data && ln -sfn ../../data/raw gemma/data/raw; }   # gemma/codec.py ищет case_metrics в gemma/data/raw

need_base() { [ -n "${GEMMA_BASE:-}" ] || { echo "GEMMA_BASE не задан" >&2; exit 2; }; }

verify() {
  sha256sum -c --quiet repro/inputs.sha256
  if [ -n "${GEMMA_BASE:-}" ]; then (cd "$GEMMA_BASE" && sha256sum -c --quiet "$OLDPWD/repro/base.sha256"); fi
  echo "verify ok"
}

fetch() {  # распаковать снимок замороженных входов в ml/
  [ -n "${SNAPSHOT:-}" ] || { echo "SNAPSHOT не задан" >&2; exit 2; }
  tar --zstd -xf "$SNAPSHOT"
}

data() {
  need_base
  # GLiNER: data/struct/{train,dev}.jsonl — ЗАМОРОЖЕННЫЙ вход (struct_data.py текущей версии даёт 17508 строк против записанных 18714)
  # Gemma: пересборка наборов; sha256 совпадает с рабочими файлами (ml-gemma/data/targeted_gen_v21_ctx3k)
  local g=gemma
  link_raw
  $PY $g/filter_gen.py --in data/gen/new_calls.jsonl --out /tmp/new_calls_clean.jsonl
  cmp /tmp/new_calls_clean.jsonl $g/data/gen/new_calls_clean.jsonl
  mkdir -p $g/synth/data && ln -sfn "$PWD/data/raw" $g/synth/data/raw
  (cd $g/synth && for v in real_only real_plus_targeted; do $PY scripts/prepare_gemma.py --variant $v --tokenizer "$GEMMA_BASE"; done)
  $PY $g/prepare_dataset.py --train $g/data/gen/new_calls_clean.jsonl --eval-frac 0 --out-dir /tmp/cfe_new_out --tokenizer "$GEMMA_BASE"
  mkdir -p $GM_DIR
  $PY - "$g" "$GM_DIR" <<'PYEOF'
import sys
g, out = sys.argv[1:]
L = lambda p: open(p, encoding="utf-8").read().splitlines()
tg = L(f"{g}/synth/data/real_plus_targeted/sft_train.jsonl")
gen = L(f"{g}/synth/data/real_only/sft_train.jsonl") + L("/tmp/cfe_new_out/sft_train.jsonl")
rows = list(dict.fromkeys(tg + gen))
open(f"{out}/sft_train.jsonl", "w", encoding="utf-8").write("\n".join(rows) + "\n")
open(f"{out}/sft_eval.jsonl", "w", encoding="utf-8").write("\n".join(L(f"{g}/synth/data/real_plus_targeted/sft_eval.jsonl")) + "\n")
print("sft_train", len(rows))
PYEOF
  echo "ожидаемые sha256: sft_train 46316203… (5402 строки), см. repro/README.md"
}

gliner() {
  $PY cfe/struct_train.py $GL_ARGS --out "$OUT"
  $PY cfe/struct_predict.py --adapter "$OUT" --out "${OUT}-v21" --splits dev,test
}

gemma() {
  need_base
  local extra="" tr="data/targeted_gen_v21_ctx3k/sft_train.jsonl" ev="data/targeted_gen_v21_ctx3k/sft_eval.jsonl"
  [ "$SMOKE" = 1 ] && extra="--smoke"
  link_raw
  mkdir -p gemma/data && [ -e gemma/data/targeted_gen_v21_ctx3k ] || ln -s "$PWD/$GM_DIR" gemma/data/targeted_gen_v21_ctx3k 2>/dev/null || true
  (cd gemma && uv sync --frozen \
    && uv run python train.py --model "$GEMMA_BASE" --train-file "$tr" --eval-file "$ev" --out adapters/$NAME --seed 42 $extra \
    && uv run python predict.py --model "$GEMMA_BASE" --adapter adapters/$NAME/adapter --data data/raw/val.jsonl \
         --out runs/$NAME/val_pred.jsonl --tag lora $([ "$SMOKE" = 1 ] && echo "--limit 8")
    if [ "$SMOKE" = 1 ]; then rm -rf adapters/repro-smoke && mv adapters/_smoke adapters/repro-smoke; mkdir -p ../runs/repro-smoke && cp -r runs/_smoke ../runs/repro-smoke/gemma; fi)
}

ensemble() { $PY cfe/ensemble.py cv; }

case "${1:-}" in
  verify) verify ;;
  fetch) fetch ;;
  data) data ;;
  gliner) gliner ;;
  gemma) gemma ;;
  ensemble) ensemble ;;
  all) verify; data; gliner; gemma; ensemble ;;
  *) echo "usage: repro.sh verify|fetch|data|gliner|gemma|ensemble|all" >&2; exit 2 ;;
esac
