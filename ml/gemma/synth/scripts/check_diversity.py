#!/usr/bin/env python3
"""Разнообразие синтетики: внутренние дубли, n-граммное покрытие, лексика.

Проверяем:
  * уникальность реплик и полных диалогов внутри синтетики;
  * долю уникальных 8-грамм;
  * разнообразие первых реплик клиента (шаблонность приветствий);
  * распределение фамилий/имён (не повторяется ли один и тот же «клиент»).

Выход: JSON в stdout (используется analyze_distribution.py и отчёты).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def shingles(text: str, n: int = 8) -> set[str]:
    toks = re.findall(r"[а-яёa-z0-9]+", text.lower())
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--compare", default=None, help="второй файл для сравнения overlap")
    a = ap.parse_args()

    calls = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    texts = [c["call_id"] for c in calls]
    full = [" ".join(t["text"] for t in c["turns"]) for c in calls]

    res = {"file": a.input, "calls": len(calls)}
    # точные дубли диалогов
    res["duplicate_dialogues"] = len(full) - len(set(full))
    # уникальные 8-граммы
    all_sh = Counter()
    for t in full:
        all_sh.update(shingles(t))
    total = sum(all_sh.values())
    res["ngram8_total"] = total
    res["ngram8_unique"] = len(all_sh)
    res["ngram8_uniqueness"] = round(len(all_sh) / max(1, total), 3)
    # самые частые 8-граммы (шаблонность)
    res["top_ngram8"] = [(g, n) for g, n in all_sh.most_common(10)]
    # разнообразие первых реплик клиента
    firsts = Counter()
    for c in calls:
        first = next((t["text"] for t in c["turns"] if t["speaker"] == "client"), "")
        firsts[first] += 1
    res["distinct_first_client_turns"] = len(firsts)
    res["top_first_client_turns"] = firsts.most_common(5)
    # разнообразие имён клиентов
    names = Counter()
    for c in calls:
        for f in c.get("facts", []):
            if f["object"]["type"] == "client":
                names[f["object"]["name"]] += 1
                break
    res["distinct_client_names"] = len(names)

    if a.compare:
        other = [json.loads(l) for l in open(a.compare, encoding="utf-8")]
        osh = set()
        for c in other:
            osh |= shingles(" ".join(t["text"] for t in c["turns"]))
        my = set()
        for t in full:
            my |= shingles(t)
        res["overlap_with_compare"] = round(len(my & osh) / max(1, len(my)), 4)
        res["compare_file"] = a.compare
        res["compare_calls"] = len(other)

    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
