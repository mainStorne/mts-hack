"""Инференс Gemma 3 1B (+ LoRA-адаптер) для кейса МТС CompactFactExtraction.

Выход — предсказания в ОФИЦИАЛЬНОЙ схеме кейса, пригодные для case_metrics.py.

Декодер устойчив к типичным сбоям генерации:
  * markdown-блоки ```json ... ```;
  * блоки <think> ... </think>;
  * обрезанный JSON (восстанавливается до последнего целого объекта);
  * компактный формат кодека {t,o,n,a,v,p,f,e} И полная официальная схема.

Постобработка по умолчанию ВЫКЛЮЧЕНА — сначала измеряем модель как есть.

Запуск:
  .venv/bin/python predict.py --adapter adapters/gemma3-1b-lora/adapter --out predictions/lora
  .venv/bin/python predict.py --out predictions/base          # без адаптера
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import codec  # noqa: E402  (общий модуль команды, только чтение)

ATTRS = {"service_state", "tariff", "ticket_status", "action", "agreement"}
OBJ_TYPES = {"client", "service", "device", "ticket", "other"}
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# --------------------------------------------------------------------------- #
# Устойчивое извлечение JSON-массива
# --------------------------------------------------------------------------- #
def _candidates(text: str):
    """Варианты подстрок, которые могут оказаться валидным JSON-массивом."""
    text = THINK_RE.sub("", text)
    m = FENCE_RE.search(text)
    if m:
        text = m.group(1)

    start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=-1)
    if start < 0:
        return []
    body = text[start:]
    end = max(body.rfind("]"), body.rfind("}"))
    if end >= 0:
        body = body[: end + 1]

    out = [body]
    # обрезанный вывод: пробуем закрыть массив
    if body.rstrip().endswith("}"):
        out.append(body.rstrip() + "]")
    # ещё агрессивнее: отрезаем до последнего целого объекта
    cut = body.rfind("}")
    while cut > 0:
        out.append(body[: cut + 1] + "]")
        cut = body.rfind("}", 0, cut)
    return out


def parse_facts_array(text: str) -> list | None:
    """JSON-массив фактов или None, если разобрать не удалось."""
    for cand in _candidates(text):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):        # модель вернула один объект вместо массива
            return [obj]
    return None


# --------------------------------------------------------------------------- #
# Приведение к официальной схеме
# --------------------------------------------------------------------------- #
def _int(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and float(x).is_integer():
        return int(x)
    if isinstance(x, str):
        try:
            return int(x.strip())
        except ValueError:
            return None
    return None


def _clean(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip()


def to_fact(obj: dict, call: dict, catalog_devices: set) -> dict | None:
    """Один элемент вывода модели -> факт в официальной схеме. None если невалиден."""
    if not isinstance(obj, dict):
        return None

    # компактный формат кодека
    if "a" in obj and "v" in obj:
        t, o, n = _int(obj.get("t")), _clean(obj.get("o") or ""), _clean(obj.get("n") or "")
        a, v = _clean(obj.get("a")), _clean(obj.get("v"))
        p, f, e = _int(obj.get("p")), _int(obj.get("f")), obj.get("e")
    else:
        # полная официальная схема
        src = obj.get("source") if isinstance(obj.get("source"), dict) else {}
        t, o = _int(src.get("turn_idx")), _clean(obj.get("o") or "")
        nobj = obj.get("object") if isinstance(obj.get("object"), dict) else {}
        n, o = _clean(nobj.get("name")), _clean(nobj.get("type") or o)
        a, v = _clean(obj.get("attribute")), _clean(obj.get("value"))
        p = _int(obj.get("polarity"))
        f = codec.day_offset(call["started_at"], obj["valid_from"]) if obj.get("valid_from") else None
        e = obj.get("valid_to")
        if e:
            e = codec.day_offset(call["started_at"], e)

    if a not in ATTRS or not v or p not in (1, -1):
        return None
    if t is None or not 0 <= t < len(call["turns"]):
        return None
    if f is None:
        return None
    e_int = _int(e) if e is not None else None
    if e is not None and e_int is None:
        return None
    if not o:
        o = "device" if (a == "service_state" and n in catalog_devices) else (
            "service" if a == "service_state" else
            "ticket" if a == "ticket_status" else "client")

    try:
        valid_from = codec.iso(call["started_at"], f)
        valid_to = None if e_int is None else codec.iso(call["started_at"], e_int)
    except (OverflowError, ValueError):
        return None

    return {
        "object": {"type": o, "name": n},
        "attribute": a,
        "value": v,
        "polarity": p,
        "speaker": call["turns"][t]["speaker"],
        "valid_from": valid_from,
        "valid_to": valid_to,
        "source": {"turn_idx": t, "span": None, "quote": None},
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / "config.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None, help="путь к LoRA-адаптеру")
    ap.add_argument("--data", type=Path, default=ROOT / "data/raw/val.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    # постобработка (по умолчанию выключена)
    ap.add_argument("--drop-valid-to", nargs="*", default=[],
                    help="атрибуты, у которых обнулить valid_to")
    ap.add_argument("--close-states", action="store_true",
                    help="применить codec.close_states")
    ap.add_argument("--max-prompt-tokens", type=int, default=3072,
                    help="лимит промпта в токенах; должен совпадать с обучением")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    model_name = a.model or cfg["model"]["name"]
    batch = a.batch or cfg["infer"]["batch_size"]
    max_new = a.max_new_tokens or cfg["infer"]["max_new_tokens"]
    tag = a.tag or ("lora" if a.adapter else "base")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_src = a.adapter or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[cfg["model"]["torch_dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, attn_implementation=cfg["model"]["attn_implementation"])
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: свободно {free/2**30:.1f} / {total/2**30:.1f} ГБ")

    calls = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if a.limit:
        calls = calls[: a.limit]
    print(f"Звонков для инференса: {len(calls)} (модель: {model_name}"
          f"{' + ' + a.adapter if a.adapter else ''})")

    catalog = json.load(open(ROOT / "data/raw/catalog.json", encoding="utf-8"))
    devices = set(catalog["devices"])

    import prepare_dataset as pd
    overhead = pd.template_overhead(tokenizer)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": codec.SYSTEM},
         {"role": "user", "content": pd.truncate_to_tokens(
             codec.render_user(c), tokenizer, a.max_prompt_tokens, overhead)}],
        tokenize=False, add_generation_prompt=True) for c in calls]

    raws, gen_time = [], 0.0
    for i in range(0, len(prompts), batch):
        enc = tokenizer(prompts[i:i + batch], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        gen_time += time.perf_counter() - t0
        raws += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        print(f"  {min(i+batch, len(prompts))}/{len(prompts)}", flush=True)

    # ---- декодирование в официальную схему -------------------------------------
    a.out.mkdir(parents=True, exist_ok=True)
    raw_path = a.out / f"{tag}_raw.jsonl"
    pred_path = a.out / f"{tag}.jsonl"
    stats = {"n": len(calls), "parsed": 0, "unparsed": 0, "facts": 0, "dropped_facts": 0}

    with open(raw_path, "w", encoding="utf-8") as fr, open(pred_path, "w", encoding="utf-8") as fp:
        for call, raw in zip(calls, raws):
            fr.write(json.dumps({"call_id": call["call_id"], "raw": raw},
                                ensure_ascii=False) + "\n")
            arr = parse_facts_array(raw)
            facts = []
            if arr is None:
                stats["unparsed"] += 1
            else:
                stats["parsed"] += 1
                for o in arr:
                    f = to_fact(o, call, devices)
                    if f is None:
                        stats["dropped_facts"] += 1
                    else:
                        facts.append(f)
            stats["facts"] += len(facts)

            if a.close_states:
                codec.close_states(facts)
            if a.drop_valid_to:
                for f in facts:
                    if f["attribute"] in a.drop_valid_to:
                        f["valid_to"] = None
            fp.write(json.dumps({"call_id": call["call_id"], "facts": facts},
                                ensure_ascii=False) + "\n")

    stats["json_valid_rate"] = round(stats["parsed"] / max(stats["n"], 1), 4)
    stats["mean_facts_per_call"] = round(stats["facts"] / max(stats["n"], 1), 2)
    stats["latency_ms_per_call"] = round(gen_time * 1000 / max(stats["n"], 1), 1)
    stats["model"] = model_name
    stats["adapter"] = a.adapter
    stats["max_new_tokens"] = max_new
    (a.out / f"{tag}_infer_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== ИНФЕРЕНС ЗАВЕРШЁН ===")
    print(f"  разобрано JSON      : {stats['parsed']}/{stats['n']} ({stats['json_valid_rate']:.1%})")
    print(f"  фактов извлечено    : {stats['facts']} (в среднем {stats['mean_facts_per_call']} на звонок)")
    print(f"  отброшено фактов    : {stats['dropped_facts']}")
    print(f"  скорость            : {stats['latency_ms_per_call']} мс/звонок")
    print(f"  предсказания        : {pred_path}")


if __name__ == "__main__":
    main()
