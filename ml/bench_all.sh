#!/usr/bin/env bash
# Все точки кривой «ресурсы × качество» -> runs/bench/<точка>/<срез>/. Запуск из ml/ на сервере с GPU: ./bench_all.sh gpu | cpu
# gpu: точки на GPU (ждёт монопольного GPU, т.к. bench.py завершается кодом 2 при чужих процессах); cpu: точки CPU-образа (cpuset, GPU не нужен).
# Уже готовые (resources.json есть) пропускаются; упавшие пишут runs/bench/<точка>/error.txt.
set -uo pipefail
cd "$(dirname "$0")"
B=runs/bench
declare -A SL=([val]=data/raw/val.jsonl [p20]=data/adv/val_p20.jsonl [p40]=data/adv/val_p40.jsonl [p60]=data/adv/val_p60.jsonl)
PY=.venv/bin/python
HOSTENV=(CFE_MODEL=runs/struct-lora CFE_PARAMS=runs/struct-lora-v21/params.json CFE_ENSEMBLE=runs/ens-v3/params.json CFE_GEMMA_MODEL=models/gemma-merged)
IMG_GPU=${IMG_GPU:-cfe-gpu:1.0-v100}

gpu_wait() { while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "$(date +%T) GPU занят, жду" >&2; sleep 120; done; }

# point slice runner... : runner — функция, печатающая команду; здесь — прямой вызов
host() {  # host POINT SLICE "ENV=.. ENV=.." bench-args...
  local p=$1 s=$2 envs=$3; shift 3
  [ -f $B/$p/$s/resources.json ] && return 0
  mkdir -p $B/$p/$s
  while :; do
    gpu_wait
    env "${HOSTENV[@]}" $envs $PY cfe/bench.py --calls "${SL[$s]}" --out $B/$p/$s "$@" > $B/$p/$s/bench.log 2>&1; rc=$?
    [ $rc -eq 2 ] && continue
    break
  done
  [ $rc -ne 0 ] && { tail -5 $B/$p/$s/bench.log > $B/$p/error.txt; echo "FAIL $p/$s"; }
  return 0
}
cont() {  # cont POINT SLICE "docker-extra-args" IMAGE bench-args...  (IMAGE с entrypoint-gpu.sh -> подкоманда bench)
  local p=$1 s=$2 dx=$3 img=$4; shift 4
  [ -f $B/$p/$s/resources.json ] && return 0
  mkdir -p $B/$p/$s
  while :; do
    gpu_wait
    docker run --rm --gpus all --user 0:0 -e HOME=/tmp -v "$PWD/runs:/app/runs" -v "$PWD/data:/app/data:ro" $dx $img bench --calls "${SL[$s]}" --out $B/$p/$s "$@" > $B/$p/$s/bench.log 2>&1; rc=$?
    [ $rc -eq 2 ] && continue
    break
  done
  [ $rc -ne 0 ] && { tail -8 $B/$p/$s/bench.log > $B/$p/error.txt; echo "FAIL $p/$s"; }
  return 0
}

case "${1:-}" in
  gpu)
    for s in val p20 p40 p60; do host gliner $s "" --mode gliner; done
    for s in val p20 p40 p60; do host gemma-b16 $s "" --mode gemma --batch 16; done
    for s in val p20 p40 p60; do host ens-hf-b16 $s "" --mode ensemble --batch 16; done
    # второй прогон val для проверки повторяемости gpu_sec_per_call (±5%)
    host ens-hf-b16-rep val "" --mode ensemble --batch 16
    host ens-hf-b1 val "" --mode ensemble --batch 1
    for s in val p20 p40; do host ens-hf-b16-ctx2048 $s CFE_GEMMA_MAX_PROMPT=2048 --mode ensemble --batch 16; done
    for s in val p20 p40; do host ens-hf-b16-ctx1536 $s CFE_GEMMA_MAX_PROMPT=1536 --mode ensemble --batch 16; done
    for s in val p20 p40; do host ens-hf-b16-new512 $s CFE_GEMMA_MAX_NEW=512 --mode ensemble --batch 16; done
    for s in val p20 p40; do cont ens-vllm-b16 $s "" $IMG_GPU --batch 16; done
    for s in val p20 p40; do cont ens-vllm-b64 $s "" $IMG_GPU --batch 64; done
    ;;
  cpu)
    # ens-cpu-t8: 8 ядер, 16 ГБ, 48 звонков по одному (ресурсы CPU); ens-cpu-full: 16 ядер, batch 4 (качество на CPU)
    p=ens-cpu-t8; mkdir -p $B/$p/val
    [ -f $B/$p/val/resources.json ] || docker run --rm --cpuset-cpus 40-47 --memory 16g -e CFE_THREADS=8 --user 0:0 -e HOME=/tmp \
      -v "$PWD/runs:/app/runs" -v "$PWD/data:/app/data:ro" --entrypoint python cfe-cpu:1.0-amd64 cfe/bench.py --calls data/raw/val.jsonl --out $B/$p/val --batch 1 --limit 48 > $B/$p/val/bench.log 2>&1 \
      || tail -5 $B/$p/val/bench.log > $B/$p/error.txt
    p=ens-cpu-full; mkdir -p $B/$p/val
    [ -f $B/$p/val/resources.json ] || timeout 28800 docker run --rm --cpuset-cpus 24-39 -e CFE_THREADS=16 --user 0:0 -e HOME=/tmp \
      -v "$PWD/runs:/app/runs" -v "$PWD/data:/app/data:ro" --entrypoint python cfe-cpu:1.0-amd64 cfe/bench.py --calls data/raw/val.jsonl --out $B/$p/val --batch 4 > $B/$p/val/bench.log 2>&1 \
      || { echo "ens-cpu-full не уложился в 8 ч или упал: первые 96 звонков" > $B/$p/note.txt; \
           docker run --rm --cpuset-cpus 24-39 -e CFE_THREADS=16 --user 0:0 -e HOME=/tmp -v "$PWD/runs:/app/runs" -v "$PWD/data:/app/data:ro" \
             --entrypoint python cfe-cpu:1.0-amd64 cfe/bench.py --calls data/raw/val.jsonl --out $B/$p/val --batch 4 --limit 96 > $B/$p/val/bench.log 2>&1 || tail -5 $B/$p/val/bench.log > $B/$p/error.txt; }
    ;;
  *) echo "usage: bench_all.sh gpu|cpu" >&2; exit 2 ;;
esac
