#!/usr/bin/env bash
# Запускает vLLM (если CFE_GEMMA_BACKEND=vllm), затем API или `bench ...`.
set -euo pipefail
cd /app
if [ "${CFE_GEMMA_BACKEND:-hf}" = "vllm" ]; then
  BENCH_SMI_BASE_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9); export BENCH_SMI_BASE_MB
  # тот же интерпретатор, что в Entrypoint базового образа (python -m vllm.entrypoints.openai.api_server).
  # V100 без bf16: vLLM запрещает float16 для gemma3 (список _FLOAT16_NOT_SUPPORTED_MODELS), а FLASH_ATTN_V100 умеет только fp16.
  # Наша дообученная модель в fp16 проверена (HF-бэкенд), поэтому на V100 (VLLM_FP16_GEMMA=1) снимаем запрет.
  ${VLLM_PYTHON:-python} -c "
import os, runpy
if os.environ.get('VLLM_FP16_GEMMA') == '1':
    import vllm.config.model as m
    m._FLOAT16_NOT_SUPPORTED_MODELS.clear()
runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
" --model /app/models/gemma-merged --served-model-name gemma \
    --dtype "${VLLM_DTYPE:-bfloat16}" --host 127.0.0.1 --port 8001 --max-model-len 4352 --max-num-seqs 64 \
    --gpu-memory-utilization "${VLLM_GPU_FRAC:-0.35}" ${VLLM_EXTRA:-} > /tmp/vllm.log 2>&1 &
  pid=$!
  for _ in $(seq 300); do
    kill -0 $pid 2>/dev/null || { echo "vLLM завершился:" >&2; tail -30 /tmp/vllm.log >&2; exit 1; }
    curl -sf http://127.0.0.1:8001/health >/dev/null && break
    sleep 1
  done
  curl -sf http://127.0.0.1:8001/health >/dev/null || { echo "vLLM не поднялся за 300 с" >&2; tail -30 /tmp/vllm.log >&2; exit 1; }
fi
if [ "${1:-}" = "bench" ]; then
  exec /app/.venv/bin/python cfe/bench.py "${@:2}"
fi
exec /app/.venv/bin/uvicorn cfe.api:app --host 0.0.0.0 --port 8000
