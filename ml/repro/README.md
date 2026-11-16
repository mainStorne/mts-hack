# Воспроизводимость обучения CFE

Полное переобучение не запускалось. Зафиксированы входы (sha256), команды и окружение; цепочка проверена короткими прогонами (`SMOKE=1 ./repro.sh all`)
и детерминированным пересчётом реранкера (`ensemble.py cv`).

| этап | команда | входы | выходы | детерминизм |
|---|---|---|---|---|
| проверка входов | `./repro.sh verify` | `repro/inputs.sha256`, `repro/base.sha256` (при `GEMMA_BASE`) | код 0/1 | точный |
| распаковка снимка | `SNAPSHOT=cfe-inputs-v1.tar.zst ./repro.sh fetch` | `/opt/mts_hack/shared/cfe-snapshot/cfe-inputs-v1.tar.zst` | файлы из `inputs.sha256` | точный |
| наборы Gemma | `GEMMA_BASE=… ./repro.sh data` | `data/raw/train.jsonl`, `data/gen/new_calls.jsonl`, `gemma/synth/data/*.jsonl` | `gemma/data/targeted_gen_v21_ctx3k/sft_{train,eval}.jsonl` (5402 / 89 строк) | точный: sha256 `sft_train` и `sft_eval` совпали с рабочими (`0ee53dfe…`, `e1c1aa6f…` для targeted; gen = real_only + новые звонки, `742f5a8b…`) |
| набор GLiNER | — (замороженный вход `data/struct/{train,dev}.jsonl`) | снимок | 18714 / dev строк | `struct_data.py` текущей версии даёт 17508 строк, а не записанные 18714: код менялся после обучения, поэтому набор не пересобирается |
| GLiNER LoRA | `./repro.sh gliner` | `data/struct`, `fastino/gliner2.5-multi-v1@7b005274…` | `runs/struct-lora-repro`, `…-v21` (пороги, сырые выходы dev/test) | нет: fp16, `deterministic: false`, seed 42 |
| Gemma LoRA | `GEMMA_BASE=… ./repro.sh gemma` | наборы Gemma, `gemma-3-1b-it` (sha256 в `base.sha256`), `gemma/uv.lock` | `gemma/adapters/…-repro`, `val_pred.jsonl` | нет: fp16, cuDNN/sdpa; seed 42, 3 эпохи, lr 2e-4, batch 4×4, r 16/α 32 (`recorded/…/run_config.json`) |
| реранкер | `./repro.sh ensemble` | `raw_test.jsonl` GLiNER, `val_pred … lora.jsonl` Gemma | `runs/ens-v3/params.json` | да: `w`, `prm` совпадают с записанными (max \|Δw\| < 1e-9) |

Побитового совпадения весов нет. Критерий повторения — метрики val в пределах ±0,01 от записанных.
`data/gen/{new_calls,repair}.jsonl` сгенерированы LLM, генератора нет — они замороженные входы и не перегенерируются.
Записанные конфиги и логи обучения: `repro/recorded/`. Окружение: `uv.lock` (GLiNER, торч 2.14.1+cu126) и `gemma/uv.lock` (Gemma).

## Проверено на чистом клоне (ноутбук Ryzen 7 5825U, 14 ГБ, без GPU)
- `./repro.sh verify` после `tar --zstd -xf cfe-inputs-v1.tar.zst`: ok; порча байта → FAILED, код 1.
- `PY=<python> GEMMA_BASE=<каталог с tokenizer.json> ./repro.sh data`: `sft_train.jsonl` (5402 строки, `46316203…`) и `sft_eval.jsonl` (`e1c1aa6f…`) совпадают с серверными побайтово. Для шага нужны `transformers`, `pyyaml`, `sentencepiece`, `jinja2` (токенайзер Gemma; веса не нужны).
- Веса из `weights/` совпадают с `MANIFEST.json` по sha256.
- GLiNER на CPU (fp32) на 20 звонках val: наборы фактов совпали с эталоном из `raw_test.jsonl` (GPU, fp16) в 20 из 20; нарушений схемы 0; 29 с на звонок.

## Повторяемость обучения на малом примере (сервер, V100, seed 42, два запуска подряд)
| модель | запуск | результат |
|---|---|---|
| GLiNER | `struct_train.py --epochs 1 --max-train 32 --seed 42` | веса адаптера **идентичны побитово** (262 тензора, max\|Δ\| = 0, sha256 `c738ee665bd1`) |
| Gemma | `train.py --max-steps 4 --batch 2 --grad-accum 1 --seed 42` | веса **различаются**: max\|Δ\| = 6,6e-4, относительное расхождение по норме 1,2e-3 (норма адаптера 31,2) |

Расхождение Gemma — недетерминизм fp16-ядер на GPU, оно накапливается с числом шагов; расхождение на полном обучении (1011 шагов) не измерялось. Поэтому критерий повторения остаётся по метрикам (val ±0,01), а не по весам. У GLiNER на этом размере детерминизм проявился несмотря на `deterministic: false` в конфиге; на полном обучении (3510 шагов) это не проверялось.
