"""Сопоставление кандидатов с эталоном теми же правилами, что и в case_metrics.py."""
import sys
import case_metrics as cm

cm.load_catalog(__file__.rsplit("/", 1)[0] + "/data/x")


def value_match(gold_value, pred_value, attr):
    vs = cm._value_sim(gold_value, pred_value)
    cg, cp = cm._canon(gold_value, attr), cm._canon(pred_value, attr)
    if cg == cp and (cg != gold_value or cp != pred_value):
        vs = 1.0
    need = cm.VALUE_SIM if min(len(cm._tokens(gold_value)), len(cm._tokens(pred_value))) <= 2 else cm.VALUE_SIM_LONG
    return vs >= need, vs


def gold_matches(gold_facts, attr, value, polarity=1):
    """индексы gold-фактов, с которыми совпало бы значение (без проверки объекта)"""
    out = []
    for gi, g in enumerate(gold_facts):
        if g["attribute"] != attr or (g.get("polarity", 1) > 0) != (polarity > 0):
            continue
        ok, vs = value_match(g["value"], value, attr)
        if ok:
            out.append((gi, vs))
    return out
