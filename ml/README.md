# ml: дообучение компактных LLM для извлечения фактов

LoRA SFT и оценка моделей, извлекающих факты из обращений в поддержку (трек МТС CompactFactExtraction).

## Почему inh-compute

Проект на сервере лежит в `/opt/mts_hack/ml` (группа `inh_shared`, доступен всем её участникам). У агента нет интерактивного доступа к Colab/Kaggle. GPU-сервер `inh-compute` (1× Tesla V100 32 ГБ) доступен по SSH: агент правит код локально, `remote.sh` синхронизирует его rsync-ом (репозиторий приватный, на сервере нет доступа к GitHub) и запускает задачи в tmux.

## Формат датасета

JSONL, одна заявка на строку (TRL «conversational prompt-completion»; loss считается только по `completion`):

```json
{"prompt": [{"role": "system", "content": "<инструкция + схема>"}, {"role": "user", "content": "<текст обращения>"}],
 "completion": [{"role": "assistant", "content": "{\"phone\": \"+7900...\", \"city\": \"Казань\"}"}]}
```

`system` необязателен. `content` в `completion` должен быть валидным JSON-объектом. Датасеты лежат в `data/` и не коммитятся.

## Команды `remote.sh`

| Команда | Действие |
|---|---|
| `sync` | rsync кода + `uv sync` на сервере |
| `data PATH...` | залить датасеты в `data/` на сервере |
| `gpu-free` | остановить `vllm-1cat` и `llama-lighton` (bge-m3 и reranker остаются) |
| `gpu-restore` | запустить их обратно |
| `run NAME CMD...` | `sync` + запуск `uv run CMD` в tmux-сессии NAME, лог в `runs/NAME.log` |
| `log NAME` | хвост лога и статус RUNNING/FINISHED |
| `pull NAME` | забрать метрики/предсказания (адаптеры и checkpoint-ы остаются на сервере) |

`gpu-free` перед обучением; `gpu-restore` после, либо когда нужна LLM на :8080 (например, разметка данных).

## Ограничения V100

- Volta (sm_70): только fp16, нет bf16 и FlashAttention-2 (используется sdpa).
- PyTorch из индекса cu126 и `< 2.15`: более новые колёса не содержат sm_70.
- Без QLoRA: fp16 LoRA моделей 0.6–4B помещается в освобождённые ~28 ГБ.

## Примеры

```bash
ml/remote.sh data my_train.jsonl my_eval.jsonl
ml/remote.sh gpu-free
ml/remote.sh run base-eval python sft/evaluate.py --data data/my_eval.jsonl --out runs/base
ml/remote.sh run train python sft/train.py --train data/my_train.jsonl --eval data/my_eval.jsonl --out runs/lora1
ml/remote.sh log train
ml/remote.sh run lora1-eval python sft/evaluate.py --adapter runs/lora1/adapter --data data/my_eval.jsonl --out runs/lora1-eval
ml/remote.sh pull lora1-eval
ml/remote.sh gpu-restore
```

## CompactFactExtraction (`cfe/`)

Компактный LoRA-экстрактор фактов (`Qwen/Qwen3.5-0.8B`) для кейса МТС: MWS. Всё запускается из `ml/`.

```bash
python3 cfe/fetch_data.py && cp data/raw/case_metrics.py cfe/   # данные + официальные метрики
ml/remote.sh sync && ml/remote.sh data data/raw                # данные на сервер
ml/remote.sh gpu-free
ml/remote.sh run cfe-all bash cfe/run_all.sh                   # prepare → train → predict ×6 → report
ml/remote.sh pull cfe-0.8b; ml/remote.sh pull zeroshot-0.8b
ml/remote.sh gpu-restore
```

Итоги: `runs/summary_val.md` (сводка + кривая «ресурсы × качество»: zero-shot, LoRA batch 8, LoRA batch 1, абляции), `runs/summary_adv.md` (adversarial-срез с ASR-шумом), `by_attribute.json` в каждом run (разбор по типам фактов).

**Цель модели** — JSON-массив фактов с полями `t` (turn_idx), `o` (тип объекта), `n` (имя), `a` (атрибут), `v` (значение), `p` (polarity), `f` (смещение valid_from в днях от дня звонка), `e` (смещение valid_to или null). `speaker` берётся из реплики `t`, `span/quote` = null.

**Детерминированные даты.** В промпт кладётся таблица смещений (`codec.date_header`: вчера/завтра/неделю и месяц назад, ближайшие прошедшие и будущие дни недели). Модель выбирает число, дату восстанавливает `codec.iso` (`started_at + f дней`). Кодек проверен oracle-прогоном на val: temporal_recall 1.0, state_accuracy 0.8957 (= gold). Постобработка: `close_states` (закрытие старых значений) и `snap_to_catalog` отключаются флагами `--no-close`/`--no-snap`.

**Методика замера ресурсов.** fp16, V100, batch 8 (и 1 для кривой), greedy, звонки отсортированы по длине промпта; GPU-секунды = сумма времени `generate` с `cuda.synchronize`, делённая на число звонков; пик памяти — `cuda.max_memory_allocated`.
