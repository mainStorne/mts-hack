# Веса модели CFE

Ансамбль v3: GLiNER2.5 (LoRA) + Gemma 3 1B (LoRA) + реранкер. Здесь лежат дообученные части; базовые модели публичные.

| файл | что | размер |
|---|---|---|
| `gemma-lora/` | LoRA-адаптер Gemma 3 1B (`gemma-v21-targeted-gen`: seed 42, 3 эпохи, lr 2e-4, batch 4×4, r 16 / α 32) | 52 МБ |
| `gliner-lora/best/` | LoRA-адаптер `fastino/gliner2.5-multi-v1@7b005274e82da37dcf1f5fd9d6a2ef413c6a004c` (seed 42, 3 эпохи) | 14 МБ |
| `gliner-params.json` | пороги GLiNER (`tau_attr`, `snap_th`) | |
| `ensemble-params.json` | реранкер и параметры сборки ансамбля (ens-v3) | |
| `MANIFEST.json` | sha256 и размеры файлов полного экспорта (`models/`), версии библиотек, число параметров | |

Слитая Gemma (`gemma-merged`, fp16, 2,0 ГБ) в git не лежит: её можно скачать из релиза GitHub `weights-v1` (файл `gemma-merged-model.safetensors`, sha256 в `MANIFEST.json`: `gemma-merged/model.safetensors`) или получить заново:

```bash
python cfe/export_models.py --gemma-base <gemma-3-1b-it> --gemma-adapter weights/gemma-lora \
  --gliner-adapter weights/gliner-lora --gliner-params weights/gliner-params.json \
  --ensemble weights/ensemble-params.json --out models
```
(`--gliner-adapter` — каталог, внутри которого `best/`.) Базовая Gemma: `google/gemma-3-1b-it`, `model.safetensors` sha256 `3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6`.
Готовые образы с весами внутри — см. `DEPLOY.md`.
