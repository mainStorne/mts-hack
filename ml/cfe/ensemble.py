#!/usr/bin/env python3
"""Ансамбль GLiNER v21 + Gemma: согласованные факты берутся всегда, «одиночки» — через реранкер (логрегрессия)."""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import struct_predict as P  # noqa: E402

T, S = P.T, P.S
import case_metrics as M  # noqa: E402

M.load_catalog(os.path.join(T.ROOT, "data/raw/catalog.json"))
import rules  # noqa: E402
import codec  # noqa: E402

vl = T.value_labels(T.load_catalog())
dm = rules.load_date_model()

GEMMA = "gemma/runs/gemma-v21-targeted-gen/val_pred.jsonl/lora.jsonl"
RAW = "runs/struct-lora-v21/raw_test.jsonl"
CALLS = P.SPLITS["test"]
V21 = "runs/struct-lora-v21/params.json"
OUT = "runs/ens-v3"
LOW_TAU = 0.3
FOLDS = 5
THETAS = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
TEMP = sorted(M.TEMPORAL_ATTRS)
RULE_CAND = [f"{a}|{k}" for a in TEMP for k in ("day1", "ago", "thru", "wd")]
DROP_CAND = ["service_state|last", "service_state|ago", "tariff|last", "tariff|ago", "ticket_status|last"]
TEMP_MARGIN = 0.02
NF = 54


def fold(call_id):
    return int(hashlib.sha256(call_id.split(":")[0].encode()).hexdigest(), 16) % FOLDS


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_inputs(calls_path, raw_path, gemma_path):
    calls = load_jsonl(calls_path)
    raws = {r["call_id"]: r for r in load_jsonl(raw_path)}
    gemma = {r["call_id"]: r for r in load_jsonl(gemma_path)}
    return calls, raws, gemma


def build_pool(calls, raws, gemma, tau=LOW_TAU, snap_th=0.5):
    gl_all = P.predict(calls, raws, vl, dm, tau, snap_th, None, keep_conf=True)
    pool = {}
    for c in calls:
        cid = c["call_id"]
        a = gl_all[cid]["facts"]
        b = gemma.get(cid, {}).get("facts", [])
        mt = M.match_facts(a, b)
        agreed = [{"ge": p, "gl": g} for g, p, _ in mt]
        mg = {id(g) for g, _, _ in mt}
        mp = {id(p) for _, p, _ in mt}
        singles = [("ge", p) for p in b if id(p) not in mp] + [("gl", g) for g in a if id(g) not in mg]
        pool[cid] = (agreed, singles)
    return pool


def _cov(value, text):
    tv = [w for w in M._tokens(value) if w != "не"]
    if not tv:
        return 0.0
    tt = M._tokens(text)
    return sum(1 for w in tv if any(M._words_close(w, u) for u in tt)) / len(tv)


def features(cid, src, p, agreed, singles, raws, call):
    attr = p["attribute"]
    x = [0.0] * NF
    x[S.ATTRS.index(attr) + (5 if src == "gl" else 0)] = 1.0
    x[10] = float(p["value"] in vl.get(attr, []))
    x[11] = float(p["speaker"] == "client")
    t = p["source"]["turn_idx"]
    x[12] = t / max(1, len(call["turns"]))
    x[13] = p.get("conf", 0.0) if src == "gl" else 0.0
    if src == "ge":
        rr = raws[cid]["raws"]
        best = 0.0
        if isinstance(t, int) and 0 <= t < len(rr):
            for item, _, _ in S._value_items(rr[t]):
                v = S._f(item)
                if v and isinstance(item, dict) and (item.get("attribute") or {}).get("label") == attr:
                    best = max(best, v[1])
        x[14] = best
    x[15] = min(sum(1 for s, q in singles if s == src and q["attribute"] == attr), 5) / 5
    x[16] = float(any(a["ge"]["attribute"] == attr and a["ge"]["value"] == p["value"] for a in agreed))
    x[17] = float(p["valid_to"] is not None)
    turns = call["turns"]
    ok = isinstance(t, int) and 0 <= t < len(turns)
    agv = [a["ge"] for a in agreed if a["ge"]["attribute"] == attr]
    oth = [q for s, q in singles if s != src and q["attribute"] == attr]
    obj = p.get("object") or {}
    f = [
        float(any(a["source"]["turn_idx"] == t for a in agv)),
        max([M._value_sim(p["value"], a["value"]) for a in agv] + [0.0]),
        max([M._value_sim(p["value"], q["value"]) for q in oth] + [0.0]),
        float(any(q["source"]["turn_idx"] == t for q in oth)),
        min(len(M._tokens(p["value"])), 12) / 12,
        _cov(p["value"], turns[t]["text"]) if ok else 0.0,
        _cov(p["value"], " ".join(u["text"] for u in turns[max(0, t - 1):t + 2])) if ok else 0.0,
        min(len(agreed), 10) / 10,
        min(len(singles), 15) / 15,
        float(bool(obj.get("name"))),
        float(obj.get("type") == "client"),
        float((p.get("valid_from") or "")[:10] == call["started_at"][:10]),
        float(len(agv) > 0),
    ]
    o = 18 + (13 if src == "gl" else 0)
    x[o:o + 13] = f
    x[44 + S.ATTRS.index(attr) + (5 if src == "gl" else 0)] = f[5]
    return x


def labels(call, agreed, singles):
    pred = [a["ge"] for a in agreed] + [p for _, p in singles]
    ok = {id(p) for _, p, _ in M.match_facts(call["facts"], pred)}
    return [1 if id(p) in ok else 0 for _, p in singles]


def sig(z):
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def fit(X, Y, epochs=1500, lr=0.5, l2=1e-3):
    n, d = len(X), len(X[0])
    w, b = [0.0] * d, 0.0
    for _ in range(epochs):
        gw, gb = [0.0] * d, 0.0
        for x, y in zip(X, Y):
            e = sig(b + sum(wi * xi for wi, xi in zip(w, x))) - y
            gb += e
            for j in range(d):
                if x[j]:
                    gw[j] += e * x[j]
        b -= lr * gb / n
        w = [wi - lr * (g / n + l2 * wi) for wi, g in zip(w, gw)]
    return w, b


def score(model, x):
    w, b = model
    return sig(b + sum(wi * xi for wi, xi in zip(w, x)))


def model_of(prm):
    """prm — содержимое params.json; возвращает (w, b) для score."""
    if len(prm["w"]) != NF:
        raise ValueError(f"params: {len(prm['w'])} weights, code expects {NF}")
    return prm["w"], prm["b"]


_PM = {}


def _mention(call, f):
    k = (call["call_id"], f["source"]["turn_idx"], f["value"], f["attribute"])
    if k not in _PM:
        _PM[k] = rules.pick_mention(call, f["source"]["turn_idx"], f["value"], f["attribute"])
    return _PM[k]


_RC = {}


def _recoverable(call):
    cid = call["call_id"]
    if cid not in _RC:
        day = date.fromisoformat(call["started_at"][:10])
        per = [M.recover_dates(t["text"], day) for t in call["turns"]]
        _RC[cid] = (day, per, set().union(*per) if per else set())
    return _RC[cid]


def assemble(call, agreed, singles, scores, prm):
    facts = []
    for a in agreed:
        f = copy.deepcopy(a["ge"])
        gl = a["gl"]
        if prm["date_src"].get(f["attribute"]) == "gl":
            f["valid_from"], f["valid_to"] = gl["valid_from"], gl["valid_to"]
        if f["source"]["turn_idx"] == gl["source"]["turn_idx"]:
            f["source"]["span"] = copy.deepcopy(gl["source"]["span"])
            f["source"]["quote"] = gl["source"]["quote"]
        facts.append(f)
    order = sorted((i for i, s in enumerate(scores) if s >= prm["theta"]), key=lambda i: -scores[i])
    for i in order:
        facts.append(copy.deepcopy(singles[i][1]))
    for f in facts:
        f.pop("conf", None)
    if prm.get("dedup", True):
        seen, out = set(), []
        for f in facts:
            if f["attribute"] == "agreement":
                if f["value"] in seen:
                    continue
                seen.add(f["value"])
            out.append(f)
        facts = out
    if prm["reclose"] == "ticket":
        tk = [f for f in facts if f["attribute"] == "ticket_status"]
        for f in tk:
            f["valid_to"] = None
        codec.close_states(tk)
    if prm.get("snap", "none") != "none":
        call_day, per_turn, union = _recoverable(call)
        for f in facts:
            if f["attribute"] not in M.TEMPORAL_ATTRS or not f.get("valid_from"):
                continue
            cur = date.fromisoformat(f["valid_from"][:10])
            if cur in union:
                continue
            ti = f["source"]["turn_idx"]
            cand = per_turn[ti] if 0 <= ti < len(per_turn) else set()
            if not cand and prm["snap"] == "any":
                cand = union
            if cand:
                best = min(cand, key=lambda d: (abs((d - cur).days), d))
                f["valid_from"] = codec.iso(call["started_at"], (best - call_day).days)
    rc, dc = set(prm.get("rule_cells", [])), set(prm.get("drop_cells", []))
    if rc or dc:
        out = []
        for f in facts:
            if f["attribute"] in M.TEMPORAL_ATTRS:
                o, k = _mention(call, f)
                cell = f"{f['attribute']}|{k}"
                if cell in rc:
                    f["valid_from"] = codec.iso(call["started_at"], o)
                if cell in dc and f["valid_to"] is not None:
                    continue
            out.append(f)
        facts = out
    return facts


def prep(calls, pool, raws):
    """Признаки и метки (метки None, если нет gold) по звонкам."""
    X, Y = {}, {}
    for c in calls:
        cid = c["call_id"]
        agreed, singles = pool[cid]
        X[cid] = [features(cid, s, p, agreed, singles, raws, c) for s, p in singles]
        Y[cid] = labels(c, agreed, singles) if "facts" in c else None
    return X, Y


def run(calls, pool, scores, prm):
    return {c["call_id"]: {"call_id": c["call_id"],
                           "facts": assemble(c, pool[c["call_id"]][0], pool[c["call_id"]][1],
                                             scores[c["call_id"]], prm)} for c in calls}


def objective(m, margin=0.0):
    n = 0
    for k in P.GATE_KEYS:
        thr = P.BASE[k] + (margin if k == "temporal_recall" else 0.0)
        n += (m.get(k) or 0) >= thr
    return (n, m["f1"], m.get("temporal_recall") or 0)


def tune(calls, pool, scores, thetas=THETAS):
    def evm(prm):
        return P.evaluate(calls, run(calls, pool, scores, prm))

    def ev(prm):
        return objective(evm(prm), TEMP_MARGIN)

    prm = {"theta": 0.35, "date_src": {a: ("gl" if a == "agreement" else "ge") for a in TEMP},
           "reclose": "none", "snap": "none", "rule_cells": [], "drop_cells": []}
    best = ev(prm)
    for rnd in range(2):
        for th in thetas:
            if th != prm["theta"]:
                q = dict(prm, theta=th)
                o = ev(q)
                if o > best:
                    prm, best = q, o
        if rnd == 1:
            break
        for a in TEMP:
            q = dict(prm, date_src=dict(prm["date_src"], **{a: "gl" if prm["date_src"][a] == "ge" else "ge"}))
            o = ev(q)
            if o > best:
                prm, best = q, o
        for v in ["none", "ticket"]:
            if v != prm["reclose"]:
                q = dict(prm, reclose=v)
                o = ev(q)
                if o > best:
                    prm, best = q, o
        for v in ("turn", "any"):
            q = dict(prm, snap=v)
            o = ev(q)
            if o > best:
                prm, best = q, o
        for cell in RULE_CAND:
            q = dict(prm, rule_cells=prm["rule_cells"] + [cell])
            o = ev(q)
            if o > best:
                prm, best = q, o
        m = evm(prm)
        goal = P.BASE["temporal_recall"] + TEMP_MARGIN
        if (m.get("temporal_recall") or 0) < goal:
            cells, cur = [], m
            while (cur.get("temporal_recall") or 0) < goal and len(cells) < len(DROP_CAND):
                opts = []
                for i, c in enumerate(DROP_CAND):
                    if c not in cells:
                        mm = evm(dict(prm, drop_cells=cells + [c]))
                        opts.append(((mm.get("temporal_recall") or 0, mm["f1"], -i), c, mm))
                _, c, cur = max(opts)
                cells.append(c)
            for c in list(cells):
                rest = [x for x in cells if x != c]
                if objective(evm(dict(prm, drop_cells=rest)), TEMP_MARGIN) > objective(cur, TEMP_MARGIN):
                    cells, cur = rest, evm(dict(prm, drop_cells=rest))
            o = objective(cur, TEMP_MARGIN)
            if o > best:
                prm, best = dict(prm, drop_cells=cells), o
        print("tune round", rnd, prm, best, flush=True)
    return prm, best


def flat(calls, X, Y):
    xs, ys = [], []
    for c in calls:
        xs += X[c["call_id"]]
        ys += Y[c["call_id"]]
    return xs, ys


def model_scores(model, calls, X):
    return {c["call_id"]: [score(model, x) for x in X[c["call_id"]]] for c in calls}


def cmd_cv(a):
    thetas = THETAS if not a.fast else [0.3, 0.35, 0.4]
    calls, raws, gemma = load_inputs(CALLS, RAW, GEMMA)
    pool = build_pool(calls, raws, gemma)
    X, Y = prep(calls, pool, raws)
    n_s = sum(len(v) for v in Y.values())
    print("agreed", sum(len(p[0]) for p in pool.values()), "singles", n_s,
          "pos", round(sum(sum(v) for v in Y.values()) / n_s, 4), flush=True)
    oof, fold_prm = {}, []
    for k in range(FOLDS):
        tr = [c for c in calls if fold(c["call_id"]) != k]
        te = [c for c in calls if fold(c["call_id"]) == k]
        model = fit(*flat(tr, X, Y))
        prm, best = tune(tr, pool, model_scores(model, tr, X), thetas)
        fold_prm.append(prm)
        oof.update(run(te, pool, model_scores(model, te, X), prm))
        print("fold", k, len(te), prm, best, flush=True)
    m_oof = P.evaluate(calls, oof)
    model = fit(*flat(calls, X, Y))
    sc = model_scores(model, calls, X)
    prm, _ = tune(calls, pool, sc, thetas)
    ins = run(calls, pool, sc, prm)
    m_in = P.evaluate(calls, ins)
    gl_prm = json.load(open(V21))
    m_gl = P.evaluate(calls, P.predict(calls, raws, vl, dm, gl_prm["tau_attr"], gl_prm["snap_th"], gl_prm.get("date_tau")))
    m_ge = P.evaluate(calls, gemma)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/pred_test.jsonl", "w", encoding="utf-8") as fh:
        for c in calls:
            fh.write(json.dumps(oof[c["call_id"]], ensure_ascii=False) + "\n")
    json.dump({"w": model[0], "b": model[1], "prm": prm, "fold_prm": fold_prm, "oof": m_oof,
               "in_sample": m_in, "gemma": GEMMA, "raw": RAW},
              open(f"{OUT}/params.json", "w"), ensure_ascii=False, indent=1)
    head = "| variant | " + " | ".join(P.COLS) + " |\n|" + "---|" * (len(P.COLS) + 1)
    rows = [P.row("baseline", P.BASE), P.row("gliner v21", m_gl), P.row("gemma", m_ge),
            P.row("ensemble OOF", m_oof), P.row("ensemble in-sample (оптимистично)", m_in)]
    ps = {k: (m_oof.get(k) or 0) >= P.BASE[k] for k in P.GATE_KEYS}
    with open(f"{OUT}/summary.md", "w", encoding="utf-8") as fh:
        fh.write(head + "\n" + "\n".join(rows) + f"\n\npasses baseline: {ps}\n\nprm final: {prm}\n\nprm по фолдам:\n")
        for k, p in enumerate(fold_prm):
            fh.write(f"- fold {k}: {p}\n")
    print(open(f"{OUT}/summary.md", encoding="utf-8").read())


def cmd_apply(a):
    calls, raws, gemma = load_inputs(a.calls, a.raw, a.gemma)
    prm_file = json.load(open(a.params))
    model = model_of(prm_file)
    pool = build_pool(calls, raws, gemma)
    X, _ = prep(calls, pool, raws)
    out = run(calls, pool, model_scores(model, calls, X), prm_file["prm"])
    with open(a.out, "w", encoding="utf-8") as fh:
        for c in calls:
            fh.write(json.dumps(out[c["call_id"]], ensure_ascii=False) + "\n")


def ensemble_call(call, rec, gemma_facts, params):
    """Один звонок: rec — сырой выход GLiNER ({"offs", "raws"}), gemma_facts — факты Gemma, params — содержимое params.json."""
    model = model_of(params)
    cid = call["call_id"]
    calls, raws = [call], {cid: rec}
    pool = build_pool(calls, raws, {cid: {"facts": gemma_facts}})
    X, _ = prep(calls, pool, raws)
    sc = model_scores(model, calls, X)
    return run(calls, pool, sc, params["prm"])[cid]["facts"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cv")
    c.add_argument("--fast", action="store_true")
    c.set_defaults(fn=cmd_cv)
    p = sp.add_parser("apply")
    for n in ("calls", "raw", "gemma", "params", "out"):
        p.add_argument("--" + n, required=True)
    p.set_defaults(fn=cmd_apply)
    a = ap.parse_args()
    a.fn(a)
