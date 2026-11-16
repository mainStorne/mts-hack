#!/usr/bin/env python3
"""Разбор ошибок по типам фактов (attribute): markdown-таблицы.

python3 cfe/error_report.py GOLD.jsonl NAME=PRED.jsonl [NAME=PRED.jsonl ...] > errors.md
Для каждой модели: n_gold/n_pred/matched, FN/FP, precision/recall, доля верных дат (temporal) и реплик-источников (turn),
полярность; для пропусков — причина: «пропущен» (в реплике-источнике нет предсказания того же attribute) или
«не то значение/объект» (в этой реплике есть предсказание того же attribute, но оно не сопоставилось).
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case_metrics as cm  # noqa: E402
import report as R  # noqa: E402

ATTRS = R.ATTRS


def f4(x):
    return "—" if x is None else f"{x:.3f}"


def missed_causes(gold, pred):
    pm = {p["call_id"]: p["facts"] for p in pred}
    cause = {a: Counter() for a in ATTRS}
    for g in gold:
        pf = pm.get(g["call_id"], [])
        pairs = cm.match_facts(g["facts"], pf)
        hit = {id(gf) for gf, _, _ in pairs}
        for gf in g["facts"]:
            if id(gf) in hit:
                continue
            ti, a = gf["source"]["turn_idx"], gf["attribute"]
            same_turn = any(p["attribute"] == a and p["source"]["turn_idx"] == ti for p in pf)
            same_attr = any(p["attribute"] == a for p in pf)
            cause[a]["не то значение/объект в той же реплике" if same_turn else ("есть факт этого типа в другой реплике" if same_attr else "в звонке нет факта этого типа")] += 1
    return cause


def main():
    gold_path, runs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    cm.load_catalog(gold_path)
    gold = R.read_jsonl(gold_path)
    for name, path in runs:
        pred = R.read_jsonl(path)
        ids = {p["call_id"] for p in pred}
        g = [x for x in gold if x["call_id"] in ids]
        att, pol = R.breakdown(g, pred)
        cause = missed_causes(g, pred)
        print(f"### {name} ({len(g)} звонков)\n")
        print("| тип факта | gold | предсказано | совпало | пропуски (FN) | лишние (FP) | precision | recall | верные даты | верная реплика |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for a, v in att.items():
            print(f"| {a} | {v['n_gold']} | {v['n_pred']} | {v['matched']} | {v['n_gold'] - v['matched']} | {v['n_pred'] - v['matched']} | "
                  f"{f4(v['precision'])} | {f4(v['recall'])} | {f4(v['temporal'])} | {f4(v['turn'])} |")
        print("\nПричины пропусков:\n")
        print("| тип факта | в звонке нет факта этого типа | есть факт этого типа в другой реплике | не то значение/объект в той же реплике |")
        print("|---|---|---|---|")
        for a in ATTRS:
            c = cause[a]
            print(f"| {a} | {c['в звонке нет факта этого типа']} | {c['есть факт этого типа в другой реплике']} | {c['не то значение/объект в той же реплике']} |")
        print("\nПолярность: " + ", ".join(f"{k}: gold {v['n_gold']}, recall {f4(v['recall'])}" for k, v in pol.items()) + "\n")


if __name__ == "__main__":
    main()
