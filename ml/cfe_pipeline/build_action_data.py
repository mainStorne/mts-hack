#!/usr/bin/env python3
"""Данные для схемы «BERT-детектор реплик + ruT5-значение» для attribute=action.

  # 1) метки реплик для BERT (turn_bert.py) и OOF tf-idf для отбора негативов ruT5
  python3 build_action_data.py detector --train raw/train_clean.jsonl --val raw/val_clean.jsonl --out ../data
  # 2) после обучения BERT: обучающие пары и реплики val для ruT5 (gen_seq.py)
  python3 build_action_data.py gen --bert-val ../runs/turn-bert/pred_val_ep0.npy --out ../data

Негативы ruT5 (цель «нет») — реплики, на которых tf-idf-классификатор (OOF по train) ошибся с p > 0.3.
"""
import argparse, json, os, pickle, random

import numpy as np

import turn_clf as T

ROW_KEYS = ("cid", "turn", "text", "prev", "next", "speaker", "y")


def prompt(r):
    return (f"Реплика ({r['speaker']}): {r['text'][:700]}\nПредыдущая: {r['prev'][:300]}\n"
            f"Следующая: {r['next'][:300]}\nДействие:")


def write(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def detector(a):
    d = os.path.join(a.out, "turn_clf")
    os.makedirs(d, exist_ok=True)
    for name, path in (("train_clean", a.train), ("val_clean", a.val)):
        rows = T.build(T.load(path))
        pickle.dump(rows, open(os.path.join(d, f"rows_{name}.pkl"), "wb"))
        write(os.path.join(d, f"{name}.jsonl"), ({k: r[k] for k in ROW_KEYS} for r in rows))
        print(name, len(rows), "позитивов", sum(r["y"] for r in rows))
    rows = pickle.load(open(os.path.join(d, "rows_train_clean.pkl"), "rb"))
    oof = T.cv(rows, C=4.0)
    np.save(os.path.join(d, "oof_train.npy"), oof)


def gen(a):
    d = os.path.join(a.out, "turn_clf")
    tr = pickle.load(open(os.path.join(d, "rows_train_clean.pkl"), "rb"))
    vr = pickle.load(open(os.path.join(d, "rows_val_clean.pkl"), "rb"))
    oof = np.load(os.path.join(d, "oof_train.npy"))
    pv = np.load(a.bert_val)
    rnd = random.Random(0)
    pos = [{"id": f"{r['cid']}|{r['turn']}", "input": prompt(r), "target": " ; ".join(r["gold_vals"])} for r in tr if r["y"]]
    negs = [r for r, p in zip(tr, oof) if not r["y"] and p > 0.3]
    rnd.shuffle(negs)
    neg = [{"id": f"{r['cid']}|{r['turn']}", "input": prompt(r), "target": "нет"} for r in negs[:a.neg]]
    g = os.path.join(a.out, "turn_gen")
    os.makedirs(g, exist_ok=True)
    write(os.path.join(g, "train.jsonl"), pos + neg)
    write(os.path.join(g, "val_flagged.jsonl"),
          ({"id": f"{r['cid']}|{r['turn']}", "input": prompt(r), "target": " ; ".join(r["gold_vals"]) or "нет", "p": float(p)}
           for r, p in zip(vr, pv) if p >= a.thr))
    print("пар обучения", len(pos), "+", len(neg), "негативов; реплик val с p >=", a.thr, ":", int((pv >= a.thr).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("detector"); x.add_argument("--train", required=True); x.add_argument("--val", required=True)
    x.add_argument("--out", default="../data")
    y = sp.add_parser("gen"); y.add_argument("--bert-val", required=True); y.add_argument("--out", default="../data")
    y.add_argument("--thr", type=float, default=0.02); y.add_argument("--neg", type=int, default=2500)
    a = ap.parse_args()
    {"detector": detector, "gen": gen}[a.cmd](a)
