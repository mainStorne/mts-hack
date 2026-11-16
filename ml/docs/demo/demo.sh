#!/usr/bin/env bash
# Демо: транскрипт -> факты. Запасной вариант без сервера: cat response.json
# Сервер: docker run --rm --gpus all -p 8000:8000 cfe-gpu:1.0-v100   (CPU: cfe-cpu:1.0-amd64)
set -euo pipefail
cd "$(dirname "$0")"
curl -s localhost:8000/health; echo
curl -s localhost:8000/extract -H 'content-type: application/json' -d @request.json | python3 -m json.tool
