#!/usr/bin/env python3
"""Сборка трёх SFT-вариантов для Gemma 3 1B: real_only / real_plus_synthetic / real_plus_targeted.

Формат примеров — ровно как в gemma3_1b/prepare_dataset.py (build_example):
codec.render_user + encode_target, completion_only_loss-совместимый формат.

Инварианты:
  * реальные данные — только train.jsonl (val/test не используются нигде);
  * синтетика и аугменты попадают ТОЛЬКО в sft_train;
  * внутренний eval собирается из реальных звонков с тем же account-хэшированием;
  * после обрезки промпта отбрасываются примеры, чей целевой JSON ссылается
    на невидимые реплики (как в gemma3_1b);
  * дубликаты call_id запрещены (аугменты уже переименованы суффиксом #augN).

Запуск:
  python3 scripts/prepare_gemma.py                    # все три варианта
  python3 scripts/prepare_gemma.py --variant real_plus_targeted
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent))

import codec  # noqa: E402
import prepare_dataset as pd  # noqa: E402  (общий модуль ml/gemma)


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def account_key(call: dict) -> str:
    return pd.account_key(call)


def build_examples(calls: list[dict], max_chars: int, tokenizer=None,
                   max_tokens: int = 0, overhead: int = 0):
    """Собрать примеры + отбросить те, где после обрезки невидимы цитируемые реплики."""
    raw = [pd.build_example(c, max_chars, tokenizer, max_tokens, overhead) for c in calls]
    out, dropped = [], 0
    for ex in raw:
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
        out.append(ex)
    return out, dropped


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/prepare_gemma.yaml"))
    ap.add_argument("--variant", default=None, help="собрать только один вариант")
    ap.add_argument("--tokenizer", default=None,
                    help="модель/путь: токенная обрезка и точные длины")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    d = cfg["data"]
    variants = cfg["variants"]
    if a.variant:
        variants = {a.variant: variants[a.variant]}

    train_calls = load_jsonl(d["train_jsonl"])
    if "val" in d["train_jsonl"]:
        raise SystemExit("Отказ: prepare_gemma.py работает только с train.jsonl")
    syn_calls = load_jsonl(ROOT / d["synthetic_pilot"])
    syn_meta = {m["call_id"]: m for m in load_jsonl(ROOT / d["synthetic_metadata"])}
    aug_calls = load_jsonl(ROOT / d["augmented_train"]) if d.get("augmented_train") else []
    targeted_fams = set(json.load(open(ROOT / d["targeted_families"]))["families"])

    max_chars = d.get("max_prompt_chars", 4200)
    max_tokens = d.get("max_prompt_tokens", 0)
    eval_frac = d.get("eval_frac", 0)

    tokenizer = None
    overhead = 0
    if a.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
        overhead = pd.template_overhead(tokenizer)
        print(f"Токенайзер: {type(tokenizer).__name__}, overhead={overhead} токенов")

    # один общий реальный eval-набор (по account-хэшу), чтобы сравнение
    # вариантов было честным
    eval_keys = set()
    for c in train_calls:
        h = int.from_bytes(hashlib.sha256(account_key(c).encode()).digest()[:8], "big")
        if h % 1000 < eval_frac * 1000:
            eval_keys.add(c["call_id"])

    for name, spec in variants.items():
        out_dir = ROOT / d["out_root"] / name
        real = [c for c in train_calls if c["call_id"] not in eval_keys] \
            if eval_frac else train_calls
        real_eval = [c for c in train_calls if c["call_id"] in eval_keys]

        extra, aug_n = [], 0
        if spec.get("synthetic"):
            if spec["synthetic"] == "all":
                extra = list(syn_calls)
            else:  # targeted
                extra = [c for c in syn_calls
                         if syn_meta.get(c["call_id"], {}).get("family") in targeted_fams]
        if spec.get("augmented"):
            aug_n = int(spec["augmented"])
            extra = extra + aug_calls[:aug_n]

        print(f"\n=== {name}: real {len(real)} + extra {len(extra)} "
              f"(synthetic {len(extra) - aug_n}, augmented {aug_n}) ===")
        tr, dr_tr = build_examples(real + extra, max_chars, tokenizer, max_tokens, overhead)
        ev, dr_ev = build_examples(real_eval, max_chars, tokenizer, max_tokens, overhead)
        print(f"  собрано train {len(tr)} (отброшено {dr_tr}), eval {len(ev)} (отброшено {dr_ev})")

        ids = [ex["call_id"] for ex in tr]
        if len(ids) != len(set(ids)):
            dup = [k for k, v in Counter(ids).items() if v > 1]
            raise SystemExit(f"Дубликаты call_id: {dup[:5]}")

        write_jsonl(out_dir / "sft_train.jsonl", tr)
        write_jsonl(out_dir / "sft_eval.jsonl", ev)

        # статистика
        tgt = [ex["completion"][0]["content"] for ex in tr]
        prompts = [len(ex["prompt"][1]["content"]) for ex in tr]
        attrs = Counter()
        facts_total = 0
        for ex in tr:
            arr = json.loads(ex["completion"][0]["content"])
            facts_total += len(arr)
            attrs.update(f["a"] for f in arr)
        stats = {
            "variant": name,
            "real_calls": len(real), "synthetic_calls": len(extra) - aug_n,
            "augmented_calls": aug_n,
            "n_train": len(tr), "n_eval": len(ev), "dropped_truncated": dr_tr,
            "facts_total": facts_total,
            "attrs": dict(attrs),
            "prompt_chars_mean": sum(prompts) / len(prompts),
            "target_chars_mean": sum(len(t) for t in tgt) / len(tgt),
            "max_prompt_chars": max_chars, "max_prompt_tokens": max_tokens,
            "provenance": {
                "real": d["train_jsonl"],
                "synthetic": str(ROOT / d["synthetic_pilot"]),
                "synthetic_selection": spec.get("synthetic", "none"),
                "augmented": str(ROOT / d["augmented_train"]) if aug_n else None,
                "augmented_meta": str(ROOT / "data/augment_metadata.jsonl") if aug_n else None,
            },
        }
        (out_dir / "dataset_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  фактов в целевом JSON: {facts_total}; атрибуты: {dict(attrs)}")
        print(f"  -> {out_dir}/sft_train.jsonl, sft_eval.jsonl, dataset_stats.json")


if __name__ == "__main__":
    main()
