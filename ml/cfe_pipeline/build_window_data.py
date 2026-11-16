#!/usr/bin/env python3
"""Оконные цели для детектора (turn_bert.py) и генератора значений (gen_seq.py) по атрибуту ATTR.

Метрика сопоставляет факты по атрибуту, значению и объекту, а реплика-источник нужна только для grounding. Поэтому
центр окна (реплика t) помечается положительным, если gold-факт атрибута стоит в репликах t-1..t+1, а цель
генератора — все gold-значения окна. Для service_state к значению добавляется имя объекта: «объект — значение».

  python3 build_window_data.py --attr service_state --train raw/train_clean.jsonl --val raw/val_clean.jsonl --out ../data/turn_win_service_state
"""
import argparse, json, os, random
from collections import defaultdict

import turn_clf as T

HEAD = {"service_state": "Состояние услуги", "tariff": "Тариф", "agreement": "Договорённость", "action": "Действие"}


def prompt(attr, r):
    return (f"{HEAD[attr]}. Реплика ({r['speaker']}): {r['text'][:600]}\nПредыдущая: {r['prev'][:400]}\n"
            f"Следующая: {r['next'][:400]}\nЗначение:")


def target(attr, f):
    return f"{f['object']['name']} — {f['value']}" if attr == "service_state" else f["value"]


def write(path, items):
    with open(path, "w", encoding="utf-8") as fh:
        for x in items:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")


def window_targets(attr, calls, rows):
    """для каждой реплики: список целевых строк gold-фактов attr в репликах t-1..t+1 (без дублей)"""
    facts = defaultdict(lambda: defaultdict(list))
    for c in calls:
        for f in c["facts"]:
            if f["attribute"] == attr and f["source"]["turn_idx"] is not None:
                facts[c["call_id"]][f["source"]["turn_idx"]].append(target(attr, f))
    out = []
    for r in rows:
        seen, ts = set(), []
        for d in (-1, 0, 1):
            for s in facts[r["cid"]].get(r["turn"] + d, []):
                if s not in seen:
                    seen.add(s)
                    ts.append(s)
        out.append(ts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", required=True, choices=["service_state", "tariff", "agreement", "action"])
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pos", type=int, default=6000); ap.add_argument("--hard", type=int, default=2000)
    ap.add_argument("--rand", type=int, default=1000)
    a = ap.parse_args()
    T.ATTR = a.attr
    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(0)
    res = {}
    for name, path in (("train", a.train), ("val", a.val)):
        calls = T.load(path)
        rows = T.build(calls)
        res[name] = (rows, window_targets(a.attr, calls, rows))
        write(os.path.join(a.out, f"{name}_clean.jsonl"),
              ({**{k: r[k] for k in ("cid", "turn", "text", "prev", "next", "speaker")}, "y": int(bool(ts))}
               for r, ts in zip(*res[name])))
        print(name, len(rows), "оконных позитивов", sum(bool(ts) for ts in res[name][1]))
    rows, ts = res["train"]
    ident = lambda r: f"{r['cid']}|{r['turn']}"
    pos = [(r, t) for r, t in zip(rows, ts) if t]
    neg = [r for r, t in zip(rows, ts) if not t]
    hard = [r for r in neg if r["n_cand"] > 0]
    rest = [r for r in neg if r["n_cand"] == 0]
    for xs in (pos, hard, rest):
        rnd.shuffle(xs)
    gen = [{"id": ident(r), "input": prompt(a.attr, r), "target": " ; ".join(t)} for r, t in pos[:a.pos]]
    gen += [{"id": ident(r), "input": prompt(a.attr, r), "target": "нет"} for r in hard[:a.hard] + rest[:a.rand]]
    write(os.path.join(a.out, "gen_train.jsonl"), gen)
    vrows, vts = res["val"]
    write(os.path.join(a.out, "gen_val_all.jsonl"),
          ({"id": ident(r), "input": prompt(a.attr, r), "target": " ; ".join(t) or "нет"} for r, t in zip(vrows, vts)))
    print("gen train", len(gen), "(позитивов", min(len(pos), a.pos), ", трудных негативов", min(len(hard), a.hard), ")")


if __name__ == "__main__":
    main()
