"""Подготовка SFT-датасета для Gemma 3 1B из train.jsonl (кейс МТС CompactFactExtraction).

Целевой формат — компактный кодек команды (cfe/codec.py):
    [{"t":7,"o":"client","n":"Елене Викторовне","a":"action","v":"...","p":1,"f":0,"e":null}]

Обоснование выбора:
  * формат проверен: codec.encode_target -> codec.decode_target даёт на официальной
    метрике precision=recall=temporal_recall=1.0, grounding_turn=1.0;
  * в ~5 раз короче полной схемы, поэтому быстрее учится и реже обрезается;
  * согласован с пайплайном команды (тем же кодеком пользуется LoRA-Qwen и постобработка GLiNER).

Никаких данных из val.jsonl здесь нет — только train.jsonl.

Запуск:
  .venv/bin/python prepare_dataset.py                       # сборка + проверки
  .venv/bin/python prepare_dataset.py --tokenizer <model>   # + точные токенные длины
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import codec  # noqa: E402  (общий модуль команды, только чтение)

ATTRS = ["service_state", "tariff", "ticket_status", "action", "agreement"]


# --------------------------------------------------------------------------- #
def load_calls(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def account_key(call: dict) -> str:
    """Ключ для честного сплита: один клиент не должен попасть в train и в eval."""
    series = call.get("series")
    if isinstance(series, dict) and series.get("account_id"):
        return str(series["account_id"])
    return call["call_id"].split(":")[0]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def template_overhead(tokenizer) -> int:
    """Токены системного промпта и служебных маркеров чат-шаблона Gemma.

    Именно они «съедают» бюджет: если мерить только user-контент, реальный
    промпт окажется длиннее и пример может быть целиком обрезан.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": codec.SYSTEM}, {"role": "user", "content": ""}],
        tokenize=False, add_generation_prompt=True)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def truncate_to_tokens(text: str, tokenizer, max_tokens: int, overhead: int) -> str:
    """Оставить заголовок с датой и столько реплик, сколько влезает в бюджет.

    Реплики отбрасываются с конца, чтобы не терять начало разговора, где обычно
    представлены клиент и тема обращения.
    """
    marker = "Диалог:\n"
    pos = text.find(marker)
    if pos < 0:
        return text
    head, body = text[: pos + len(marker)], text[pos + len(marker):]

    head_tokens = len(tokenizer(head, add_special_tokens=False)["input_ids"])
    if head_tokens + overhead >= max_tokens:
        return head  # даже заголовок не влезает — берём только его

    lines = [l for l in body.split("\n") if l.strip()]
    kept, budget = [], max_tokens - overhead - head_tokens
    for line in lines:
        n = len(tokenizer(line, add_special_tokens=False)["input_ids"])
        if budget - n < 0:
            break
        kept.append(line)
        budget -= n

    # BPE может склеивать токены на стыках, поэтому проверяем итоговую длину
    # и, если она вылезла за лимит, отбрасываем последние реплики.
    while kept and (overhead + len(tokenizer(head + "\n".join(kept),
                                              add_special_tokens=False)["input_ids"])) > max_tokens:
        kept.pop()
    return head + "\n".join(kept)


def build_example(call: dict, max_prompt_chars: int, tokenizer=None,
                  max_prompt_tokens: int = 0, overhead: int = 0) -> dict:
    """Собрать пример. Обрезка — по токенам, если доступен токенайзер.

    Почему по токенам: русский ASR-текст токенизуется в ~1.4 токена на символ,
    поэтому обрезка по символам не ограничивает длину (замерено: до 8346 токенов
    промпта при лимите 6000 символов). Если обрезать только на стороне trainer,
    целевой JSON начнёт ссылаться на реплики, которых модель не видит.
    """
    user = codec.render_user(call)
    if tokenizer is not None and max_prompt_tokens:
        user = truncate_to_tokens(user, tokenizer, max_prompt_tokens, overhead)
    elif max_prompt_chars:
        user = user[:max_prompt_chars]

    prompt = [
        {"role": "system", "content": codec.SYSTEM},
        {"role": "user", "content": user},
    ]
    # Формат — как в проверенном cfe/prepare.py команды: оба поля являются
    # списками словарей-сообщений, поэтому is_conversational() в TRL 1.18
    # детерминированно возвращает True и completion_only_loss корректно маскирует промпт.
    return {
        "call_id": call["call_id"],
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": codec.encode_target(call)}],
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / "config.yaml")
    ap.add_argument("--train", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-prompt-chars", type=int, default=None)
    ap.add_argument("--max-prompt-tokens", type=int, default=None,
                    help="лимит промпта в токенах (требует --tokenizer); "
                         "предпочтительнее символьной обрезки")
    ap.add_argument("--eval-frac", type=float, default=None)
    ap.add_argument("--tokenizer", default=None,
                    help="имя модели или путь: посчитать точные токенные длины "
                         "и включить токенную обрезку")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.max_prompt_tokens is None:
        a.max_prompt_tokens = cfg["data"].get("max_prompt_tokens", 0)
    train_path = a.train or Path(cfg["data"]["train_jsonl"])
    out_dir = (a.out_dir or ROOT / cfg["data"]["out_dir"]).resolve()
    max_chars = a.max_prompt_chars if a.max_prompt_chars is not None else cfg["data"].get("max_prompt_chars", 0)
    eval_frac = a.eval_frac if a.eval_frac is not None else cfg["data"]["eval_frac"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
        print(f"Токенайзер: {type(tokenizer).__name__}, vocab={tokenizer.vocab_size}")

    # ---- защита от случайного использования val ---------------------------------
    if "val" in train_path.name:
        raise SystemExit("Отказ: prepare_dataset.py должен работать только с train.jsonl")

    calls = load_calls(train_path)
    gold_by_id = {c["call_id"]: c for c in calls}
    print(f"Загружено звонков из {train_path}: {len(calls)}")

    # ---- сборка примеров ---------------------------------------------------------
    overhead = template_overhead(tokenizer) if tokenizer is not None else 0
    if overhead:
        print(f"Накладные токены шаблона (system + служебные): {overhead}")
    raw_examples = [build_example(c, max_chars, tokenizer, a.max_prompt_tokens, overhead)
                    for c in calls]
    print(f"Примеров собрано: {len(raw_examples)}")

    # Отбрасываем примеры, где после обрезки целевой JSON ссылается на реплику,
    # которой модель не видит: иначе она учится предсказывать turn_idx «вслепую».
    examples, dropped = [], 0
    for ex in raw_examples:
        idxs = [int(m) for m in re.findall(r"^\[(\d+)\]", ex["prompt"][1]["content"], re.M)]
        last_turn = max(idxs) if idxs else -1
        try:
            arr = json.loads(ex["completion"][0]["content"])
        except Exception:
            dropped += 1
            continue
        if any(f["t"] > last_turn for f in arr):
            dropped += 1
            continue
        examples.append(ex)
    if dropped:
        print(f"Отброшено примеров с невидимыми репликами: {dropped}")
    if tokenizer is not None and a.max_prompt_tokens:
        print(f"Промпт обрезан по {a.max_prompt_tokens} токенам")

    # ---- сплит train/eval по account_id ------------------------------------------
    train_rows, eval_rows = [], []
    if eval_frac > 0:
        for ex in examples:
            h = int.from_bytes(
                hashlib.sha256(account_key(gold_by_id[ex["call_id"]]).encode()).digest()[:8], "big")
            (eval_rows if h % 1000 < eval_frac * 1000 else train_rows).append(ex)
    else:
        train_rows = examples
    print(f"train: {len(train_rows)}  eval (только для мониторинга): {len(eval_rows)}")

    write_jsonl(out_dir / "sft_train.jsonl", train_rows)
    write_jsonl(out_dir / "sft_eval.jsonl", eval_rows)

    # ---- проверки качества -------------------------------------------------------
    attr_c = Counter()
    pol_c = Counter()
    facts_per_call = []
    bad_json = 0
    prompt_chars, target_chars = [], []
    seen_ids = set()

    for ex in examples:
        call = gold_by_id[ex["call_id"]]
        if ex["call_id"] in seen_ids:
            raise SystemExit(f"Дубликат call_id: {ex['call_id']}")
        seen_ids.add(ex["call_id"])

        tgt = ex["completion"][0]["content"]
        target_chars.append(len(tgt))
        prompt_chars.append(len(ex["prompt"][1]["content"]))
        try:
            arr = json.loads(tgt)
            assert isinstance(arr, list)
        except Exception:
            bad_json += 1
            continue

        facts_per_call.append(len(arr))
        for f in arr:
            attr_c[f["a"]] += 1
            pol_c[f["p"]] += 1
            # целевой JSON должен полностью описывать золотой факт
            cands = [x for x in call["facts"] if x["source"]["turn_idx"] == f["t"]
                     and x["attribute"] == f["a"] and x["value"] == f["v"]]
            assert any(f["f"] == (date.fromisoformat(g["valid_from"][:10])
                                  - date.fromisoformat(call["started_at"][:10])).days
                       and f["p"] == g["polarity"] for g in cands), ex["call_id"]

    n = len(examples)
    print("\n=== ПРОВЕРКИ КАЧЕСТВА ===")
    print(f"  примеров                 : {n}")
    print(f"  невалидный целевой JSON  : {bad_json}")
    print(f"  фактов на звонок         : mean={sum(facts_per_call)/n:.2f} "
          f"max={max(facts_per_call)} total={sum(facts_per_call)}")
    print(f"  атрибуты                 : {dict(attr_c)}")
    print(f"  полярность               : {dict(pol_c)}")
    print(f"  промпт, символов         : mean={sum(prompt_chars)/n:.0f} "
          f"p90={sorted(prompt_chars)[int(0.9*n)]} max={max(prompt_chars)}")
    print(f"  ответ, символов          : mean={sum(target_chars)/n:.0f} "
          f"p90={sorted(target_chars)[int(0.9*n)]} max={max(target_chars)}")
    if tokenizer is not None and a.max_prompt_tokens:
        truncated = sum(
            1 for c in calls
            if len(tokenizer(codec.render_user(c), add_special_tokens=False)["input_ids"])
            > a.max_prompt_tokens)
        print(f"  звонков с обрезанным промптом: {truncated} ({truncated/n:.3f})")

    # ---- точные токенные длины -----------------------------------------------------
    if tokenizer is not None:
        lens = sorted(len(tokenizer(ex["prompt"][1]["content"],
                                    add_special_tokens=False)["input_ids"])
                      for ex in examples)
        print(f"\n  токенов промпта: mean={sum(lens)/n:.0f} "
              f"p90={lens[int(0.9*n)]} p99={lens[int(0.99*n)]} max={max(lens)}")
        # ФАКТИЧЕСКАЯ длина: полный промпт (system + user + служебные токены
        # шаблона) плюс целевой ответ. Именно она сравнивается с max_length,
        # и если ответ не влезает, TRL молча отбросит пример как fully masked.
        full_prompt, target = [], []
        for ex in examples:
            text = tokenizer.apply_chat_template(ex["prompt"], tokenize=False,
                                                 add_generation_prompt=True)
            full_prompt.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
            target.append(len(tokenizer(ex["completion"][0]["content"],
                                        add_special_tokens=False)["input_ids"]))
        full_prompt.sort(); target.sort()
        total = sorted(a_ + b_ for a_, b_ in zip(full_prompt, target))
        print(f"  полный промпт (с шаблоном): mean={sum(full_prompt)/n:.0f} "
              f"p99={full_prompt[int(0.99*n)]} max={max(full_prompt)}")
        print(f"  ответ: mean={sum(target)/n} p99={target[int(0.99*n)]} max={max(target)}")
        print(f"  промпт+ответ: p99={total[int(0.99*n)]} max={max(total)}")
        for lim in (2048, 3072, 4096):
            over = sum(1 for x in total if x > lim)
            print(f"    примеров длиннее {lim}: {over} ({over/n:.3%})")
        rec = {"tokenizer": a.tokenizer, "max_prompt_tokens": a.max_prompt_tokens,
               "overhead_tokens": overhead,
               "user_content_tokens": {"mean": sum(lens)/n, "p90": lens[int(0.9*n)],
                                       "p99": lens[int(0.99*n)], "max": max(lens)},
               "full_prompt_tokens": {"mean": sum(full_prompt)/n,
                                      "p99": full_prompt[int(0.99*n)], "max": max(full_prompt)},
               "target_tokens": {"mean": sum(target)/n, "p99": target[int(0.99*n)],
                                 "max": max(target)},
               "total_tokens": {"p99": total[int(0.99*n)], "max": max(total)},
               "recommended_max_length": max(2048, total[int(0.99*n)] + 200)}
        (out_dir / "length_stats.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  рекомендуемый max_length: {rec['recommended_max_length']}")

    stats = {
        "source": str(train_path), "n_examples": n, "n_train": len(train_rows),
        "n_eval": len(eval_rows), "bad_json": bad_json,
        "attrs": dict(attr_c), "polarity": {str(k): v for k, v in pol_c.items()},
        "facts_per_call_mean": sum(facts_per_call) / n,
        "prompt_chars_mean": sum(prompt_chars) / n,
        "target_chars_mean": sum(target_chars) / n,
        "truncated_calls": truncated, "max_prompt_chars": max_chars,
        "max_prompt_tokens": a.max_prompt_tokens,
        "target_format": "compact codec (cfe/codec.py encode_target)",
    }
    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nДатасет сохранён в {out_dir}/sft_train.jsonl и sft_eval.jsonl")


if __name__ == "__main__":
    main()
