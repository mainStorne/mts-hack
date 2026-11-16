#!/usr/bin/env python3
"""Классификатор «в реплике есть действие» (attribute=action).

Постановка: объект — реплика, метка 1, если у какого-то gold-факта action source.turn_idx == idx реплики.
Шаг 1 пайплайна: этот классификатор поднимает recall по репликам; шаг 2 (action_gen.py) — значение
только для отмеченных реплик.

  python3 turn_clf.py cv   raw/train_clean.jsonl            # OOF по train (GroupKFold по звонкам)
  python3 turn_clf.py eval raw/train_clean.jsonl raw/val_clean.jsonl
  python3 turn_clf.py fit  raw/train_clean.jsonl -o turn_clf.pkl
"""
import argparse, json, pickle, re, sys
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, precision_recall_curve

from candidates import CandidateGen
from features import FAIL

HERE = __file__.rsplit("/", 1)[0] or "."
import os
ATTR = os.environ.get("ATTR", "action")
_GEN = None


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _gen():
    global _GEN
    if _GEN is None:
        _GEN = CandidateGen(f"{HERE}/data/catalog.json")
    return _GEN


def _call_rows(d):
    """строки по репликам: текст, соседи, спикер, признаки кандидатов-действий, метка"""
    cands, _, _, _ = _gen().call(d)
    by_turn, cl = {}, {}
    for c in cands:
        if c["attr"] == ATTR:
            by_turn.setdefault(c["turn"], []).append(c["score"])
            cl.setdefault(c["turn"], []).append((c["value"], c["score"], c["start"], c["end"]))
    gold = {f["source"]["turn_idx"] for f in d.get("facts", []) if f["attribute"] == ATTR}
    gv = {}
    for f in d.get("facts", []):
        if f["attribute"] == ATTR:
            gv.setdefault(f["source"]["turn_idx"], []).append(f["value"])
    turns = d["turns"]; n = len(turns)
    rows = []
    for i, t in enumerate(turns):
        sc = by_turn.get(i, [])
        rows.append(dict(
            cid=d["call_id"], turn=i, text=t["text"],
            prev=turns[i - 1]["text"] if i else "", next=turns[i + 1]["text"] if i + 1 < n else "",
            speaker=t["speaker"] or "?", relpos=i / max(1, n - 1),
            n_cand=len(sc), max_score=max(sc, default=0.0),
            fail=int(bool(FAIL.search(t["text"].lower()))),
            cands=cl.get(i, []), gold_vals=gv.get(i, []), y=int(i in gold)))
    return rows


def build(calls, workers=8):
    with Pool(workers) as p:
        return [r for rs in p.map(_call_rows, calls, chunksize=20) for r in rs]


class Featurizer:
    def __init__(self):
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200000)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, max_features=300000)
        self.nb = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True, max_features=50000)

    @staticmethod
    def dense(rows):
        return sp.csr_matrix(np.array([[
            r["speaker"] == "client", r["speaker"] == "operator", r["relpos"], r["n_cand"] > 0,
            min(r["n_cand"], 3) / 3, r["max_score"], r["fail"], np.log1p(len(r["text"])) / 6,
            "?" in r["text"],
        ] for r in rows], dtype=np.float32))

    def fit_transform(self, rows):
        return sp.hstack([self.w.fit_transform([r["text"] for r in rows]),
                          self.c.fit_transform([r["text"] for r in rows]),
                          self.nb.fit_transform([r["prev"] for r in rows]),
                          self.nb.transform([r["next"] for r in rows]),
                          self.dense(rows)]).tocsr()

    def transform(self, rows):
        return sp.hstack([self.w.transform([r["text"] for r in rows]),
                          self.c.transform([r["text"] for r in rows]),
                          self.nb.transform([r["prev"] for r in rows]),
                          self.nb.transform([r["next"] for r in rows]),
                          self.dense(rows)]).tocsr()


def fit_model(rows, C):
    f = Featurizer()
    X = f.fit_transform(rows)
    y = np.array([r["y"] for r in rows])
    return f, LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(X, y)


def report(name, y, p, base_pred=None):
    ap = average_precision_score(y, p)
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    k = int(np.argmax(f1[:-1]))
    print(f"[{name}] n={len(y)} pos={int(y.sum())} AP={ap:.4f} bestF1={f1[k]:.4f} @thr={th[k]:.3f} (P={pr[k]:.3f} R={rc[k]:.3f})")
    for target in (0.80, 0.85, 0.90, 0.95):
        ok = np.where(rc[:-1] >= target)[0]
        if len(ok):
            j = ok[-1]
            print(f"    recall>={target:.2f}: thr={th[j]:.3f} P={pr[j]:.3f} R={rc[j]:.3f}")
    if base_pred is not None:
        tp = int((base_pred & (y == 1)).sum())
        print(f"    baseline «есть кандидат-действие»: P={tp / max(1, base_pred.sum()):.3f} R={tp / y.sum():.3f}")


def cv(rows, C, folds=5):
    y = np.array([r["y"] for r in rows]); groups = [r["cid"] for r in rows]
    oof = np.zeros(len(rows))
    for k, (tr, te) in enumerate(GroupKFold(folds).split(rows, y, groups)):
        f, m = fit_model([rows[i] for i in tr], C)
        oof[te] = m.predict_proba(f.transform([rows[i] for i in te]))[:, 1]
        print("fold", k, "AP", round(average_precision_score(y[te], oof[te]), 4), file=sys.stderr)
    report(f"OOF train C={C}", y, oof, np.array([r["n_cand"] > 0 for r in rows]))
    return oof


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "eval", "fit"])
    ap.add_argument("train"); ap.add_argument("val", nargs="?")
    ap.add_argument("-o", default="turn_clf.pkl"); ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--cache", default="/tmp/turn_rows_" + ATTR + "_{}.pkl")
    a = ap.parse_args()

    def rows_of(path):
        cp = a.cache.format(re.sub(r"\W", "_", path))
        try:
            return pickle.load(open(cp, "rb"))
        except FileNotFoundError:
            r = build(load(path)); pickle.dump(r, open(cp, "wb")); return r

    rows = rows_of(a.train)
    if a.cmd == "cv":
        oof = cv(rows, a.C)
        pickle.dump(oof, open(f"/tmp/turn_oof_{ATTR}.pkl", "wb"))
    elif a.cmd == "eval":
        f, m = fit_model(rows, a.C)
        vr = rows_of(a.val)
        y = np.array([r["y"] for r in vr])
        report("val", y, m.predict_proba(f.transform(vr))[:, 1], np.array([r["n_cand"] > 0 for r in vr]))
    else:
        f, m = fit_model(rows, a.C)
        pickle.dump({"feat": f, "clf": m}, open(a.o, "wb"))
