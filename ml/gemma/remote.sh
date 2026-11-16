#!/usr/bin/env bash
set -euo pipefail

HOST=inh-compute
REMOTE="${GEMMA_REMOTE:-/opt/mts_hack/ml-gemma}"
LOCAL="$(cd "$(dirname "$0")" && pwd)"
UVENV="export UV_PYTHON_INSTALL_DIR=/opt/mts_hack/.uv-python UV_LINK_MODE=copy; umask 002;"
USAGE='usage: remote.sh sync|data|gpu-free|gpu-restore|run NAME CMD...|log NAME|pull NAME'
GPU_Q='nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'

sync_code() {
  ssh "$HOST" "mkdir -p $REMOTE"
  # /data, /adapters, /runs, /synth/data/* (generated) живут только на сервере
  rsync -rlzt --delete --chmod=Dg+rwxs,Fg+rw \
    --exclude .venv --exclude __pycache__ --exclude /uv.lock \
    --exclude /data --exclude /adapters --exclude /runs \
    --filter 'P /synth/data/**' "$LOCAL/" "$HOST:$REMOTE/"
  ssh "$HOST" "$UVENV cd $REMOTE && uv sync"
}

cmd="${1:-}"
[ $# -gt 0 ] && shift
case "$cmd" in
  sync) sync_code ;;
  data)
    ssh "$HOST" "mkdir -p $REMOTE/data"
    rsync -rlzt --chmod=Dg+rwxs,Fg+rw "${@:-$LOCAL/../data/raw}" "$HOST:$REMOTE/data/"
    ;;
  gpu-free)
    ssh "$HOST" "docker stop vllm-1cat; ~/llm/run-lighton.sh stop || true; $GPU_Q"
    ;;
  gpu-restore)
    ssh "$HOST" "docker start vllm-1cat; ~/llm/run-lighton.sh start; $GPU_Q"
    ;;
  run)
    name="$1"; shift
    sync_code
    quoted="$(printf '%q ' "$@")"
    ssh "$HOST" "cd $REMOTE && mkdir -p runs && tmux new-session -d -s $name \"$UVENV uv run $quoted 2>&1 | tee runs/$name.log\""
    ;;
  log)
    name="$1"
    ssh "$HOST" "tail -n 50 $REMOTE/runs/$name.log; tmux has-session -t $name 2>/dev/null && echo RUNNING || echo FINISHED"
    ;;
  pull)
    name="$1"
    mkdir -p "$LOCAL/runs/$name"
    rsync -az --exclude 'checkpoint-*' --exclude adapter "$HOST:$REMOTE/runs/$name/" "$LOCAL/runs/$name/"
    ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
