# CFE: развёртывание

Сервис принимает транскрипт с метаданными и возвращает факты в схеме кейса (`data/raw/schema.md` §2).
Модель — ансамбль GLiNER2.5 (LoRA) + Gemma 3 1B (LoRA, слитая) + реранкер (`models/ensemble-params.json`).

## Что поставляется
| артефакт | где |
|---|---|
| веса и конфигурация инференса | `models/` (`gemma-merged/`, `gliner-lora/`, `hf/` — база GLiNER, `gliner-params.json`, `ensemble-params.json`, `MANIFEST.json` с sha256 и версиями) |
| образ CPU (amd64) | `cfe-cpu-1.0-amd64.tar.gz` (`docker save`), веса внутри. arm64 не собран: на сервере нельзя включить QEMU (binfmt не регистрируется без root) |
| образ GPU | `cfe-gpu:1.0-v100` (1Cat vLLM для V100), `cfe-gpu:1.0` (обычный GPU, vLLM) |
| код обучения, сиды | `cfe/struct_train.py` (seed 42), `gemma/train.py` (seed 42), `repro.sh`, `repro/README.md` |

## Запуск образа
```bash
# CPU (Linux/amd64; на Mac с Apple Silicon — тот же образ под эмуляцией: --platform linux/amd64, заметно медленнее)
docker load -i cfe-cpu-1.0-amd64.tar.gz
docker run --rm --platform linux/amd64 -p 8000:8000 cfe-cpu:1.0-amd64
# GPU
docker run --rm --gpus all -p 8000:8000 cfe-gpu:1.0-v100      # V100
docker run --rm --gpus all -p 8000:8000 cfe-gpu:1.0           # A100/H100 и др.
```
Сеть не нужна. Первый ответ CPU-образа — после загрузки моделей (десятки секунд). Ресурсы: CPU-образ ≥ 8 ГБ RAM (рекомендуется 16 ГБ, 8 ядер); GPU-образ ≈ 0,35 доли памяти GPU под vLLM + GLiNER.

Нативный arm64 (Mac) — сборка на самой машине: нужен каталог `models/` из поставки; команда: `docker buildx build --platform linux/arm64 -f docker/Dockerfile.cpu -t cfe-cpu:1.0-arm64 --load .` из каталога `ml/`; не проверялось.

## HTTP-интерфейс
- `GET /health` — статус, устройство, бэкенд Gemma.
- `POST /extract` — тело `{"call_id", "started_at": ISO, "turns": [{"idx"?, "speaker": "client|operator", "text"}]}` → `{"call_id", "facts": [...]}`.
- `POST /extract_batch` — массив звонков (до 64), ответ в том же порядке.
- Схема факта: `object{type,name}`, `attribute`, `value`, `polarity`, `speaker`, `valid_from`, `valid_to`, `source{turn_idx,span,quote}`. OpenAPI — `GET /docs`.

```bash
curl -s localhost:8000/extract -H 'content-type: application/json' -d '{"call_id":"c1","started_at":"2026-05-04T10:00:00","turns":[{"speaker":"client","text":"Интернет пропал со вторника"},{"speaker":"operator","text":"Оформляю заявку"}]}'
```

## Переменные окружения (см. `cfe/runtime.py`)
`CFE_DEVICE` (cpu|cuda), `CFE_GEMMA_BACKEND` (hf|vllm), `CFE_GEMMA_BATCH`, `CFE_GEMMA_MAX_PROMPT`, `CFE_GEMMA_MAX_NEW`, `CFE_THREADS`, `CFE_MODEL`, `CFE_PARAMS`, `CFE_ENSEMBLE`, `CFE_GEMMA_MODEL`, `VLLM_GPU_FRAC`.

## Без Docker
`uv sync --frozen`, затем `CFE_MODEL=models/gliner-lora CFE_PARAMS=models/gliner-params.json CFE_ENSEMBLE=models/ensemble-params.json CFE_GEMMA_MODEL=models/gemma-merged HF_HOME=models/hf HF_HUB_OFFLINE=1 uv run uvicorn cfe.api:app --port 8000`.

## Замер и воспроизведение
- Ресурсы и качество: `cfe/bench.py`; кривая «ресурсы × качество» — `runs/bench/curve.md`.
- Проверка схемы: `python3 cfe/validate_schema.py CALLS.jsonl PRED.jsonl`.
- Обучение и проверка входов: `repro.sh`, `repro/README.md` (сиды 42; побитового совпадения нет, критерий — метрики val ±0,01).
- Экспорт весов из адаптеров: `cfe/export_models.py`.
