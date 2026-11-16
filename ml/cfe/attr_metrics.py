#!/usr/bin/env python3
"""Метрики case_metrics по каждому атрибуту: python3 cfe/attr_metrics.py data/raw/val.jsonl runs/<eval>/pred_test.jsonl"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import case_metrics  # noqa: E402
case_metrics.load_catalog(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "catalog.json"))
import struct_schema as S  # noqa: E402

COLS = ["n_gold_facts", "n_pred_facts", "fact_precision", "fact_recall", "temporal_recall",
        "state_accuracy", "grounding_turn", "grounding_span"]


def load(path):
    return {c["call_id"]: c for c in map(json.loads, open(path, encoding="utf-8"))}


def by_attr(gold, pred, attr):
    sel = lambda cs: {k: {**c, "facts": [f for f in c["facts"] if f["attribute"] == attr]} for k, c in cs.items()}
    return case_metrics.evaluate(sel(gold), sel(pred))


def report(gold, pred):
    rows = ["| attr | " + " | ".join(COLS) + " | f1 |", "|" + "---|" * (len(COLS) + 2)]
    for a in S.ATTRS + [None]:
        m = by_attr(gold, pred, a) if a else case_metrics.evaluate(gold, pred)
        p, r = m["fact_precision"], m["fact_recall"]
        f1 = round(2 * p * r / (p + r), 4) if p + r else 0.0
        rows.append(f"| {a or 'ALL'} | " + " | ".join("" if m.get(k) is None else str(m[k]) for k in COLS) + f" | {f1} |")
    return "\n".join(rows)


if __name__ == "__main__":
    print(report(load(sys.argv[1]), load(sys.argv[2])))
