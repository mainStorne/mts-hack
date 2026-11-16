#!/usr/bin/env python3
"""Флайвил разметки: GLiNER + Gemma -> silver.jsonl + review_queue.jsonl (+ eval.md при наличии gold).

Silver — не золото: в обучение только отдельным источником с пометкой `label_source`.
Скрипт не пишет в data/ и не объединяет silver с gold.
"""
import argparse
import collections
import copy
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ensemble as E  # noqa: E402

M, S = E.M, E.S
CONFIRM = re.compile(r"\b(да|хорошо|давайте|согласен|согласна|ладно|подходит|угу|договорились|оформляйте|устраивает|ок|окей|спасибо)\b", re.I)


def _iso(s):
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def validate_fact(f, call):
    e = []
    o = f.get("object")
    if not isinstance(o, dict) or o.get("type") not in S.TYPES or not isinstance(o.get("name"), str):
        e.append("object")
    if f.get("attribute") not in S.ATTRS:
        e.append("attribute")
    if not isinstance(f.get("value"), str) or not f["value"]:
        e.append("value")
    if f.get("polarity") not in (1, -1):
        e.append("polarity")
    if f.get("speaker") not in ("client", "operator"):
        e.append("speaker")
    vf = _iso(f.get("valid_from"))
    if vf is None:
        e.append("valid_from")
    if f.get("valid_to") is not None:
        vt = _iso(f["valid_to"])
        if vt is None or (vf is not None and vt.date() < vf.date()):
            e.append("valid_to")
    src = f.get("source")
    if not isinstance(src, dict):
        return e + ["source"]
    t = src.get("turn_idx")
    if not isinstance(t, int) or isinstance(t, bool) or not 0 <= t < len(call["turns"]):
        return e + ["turn_idx"]
    n = len(call["turns"][t]["text"])
    sp = src.get("span")
    if sp is not None and not (isinstance(sp, (list, tuple)) and len(sp) == 2
                               and all(isinstance(x, int) for x in sp) and 0 <= sp[0] < sp[1] <= n):
        e.append("span")
    if src.get("quote") is not None and not isinstance(src["quote"], str):
        e.append("quote")
    return e


def grounded(f, call):
    turn = call["turns"][f["source"]["turn_idx"]]
    if f["speaker"] != turn["speaker"]:
        return False
    text, sp = turn["text"], f["source"].get("span")
    if sp is not None:
        return text[sp[0]:sp[1]] == f["source"].get("quote")
    q = f["value"].split(" — ")[0] if f["attribute"] == "action" else f["value"]
    return S.locate_value(text, q)[0] >= S.TRAIN_MIN_SIM


def _day(s):
    return s[:10] if s else None


def pair_checks(ge, gl, call):
    r = []
    if ge["object"]["type"] != gl["object"]["type"]:
        r.append("object_type")
    if ge["source"]["turn_idx"] != gl["source"]["turn_idx"]:
        r.append("source_turn")
    if ge["polarity"] != gl["polarity"]:
        r.append("polarity")
    if not grounded(gl, call):
        r.append("ungrounded")
    if gl["attribute"] == "agreement" and gl["speaker"] == "operator":
        t = gl["source"]["turn_idx"]
        nxt = call["turns"][t + 1:t + 3]
        if not any(x["speaker"] == "client" and CONFIRM.search(x["text"]) for x in nxt):
            r.append("operator_unconfirmed")
    ta = _day(ge["valid_from"]) == _day(gl["valid_from"]) and _day(ge["valid_to"]) == _day(gl["valid_to"])
    return r, ta


def canonical(ge, gl):
    c = copy.deepcopy(ge)
    a = c["attribute"]
    if c["value"] not in E.vl.get(a, []) and gl["value"] in E.vl.get(a, []):
        c["value"] = gl["value"]
    c["source"]["span"] = gl["source"]["span"]
    c["source"]["quote"] = gl["source"]["quote"]
    c.pop("conf", None)
    return c


def side(f):
    return {"value": f["value"], "object": f["object"], "polarity": f["polarity"],
            "valid_from": f["valid_from"], "valid_to": f["valid_to"], "turn_idx": f["source"]["turn_idx"],
            "span": f["source"].get("span"), "quote": f["source"].get("quote")}


def strip(f):
    f = copy.deepcopy(f)
    f.pop("conf", None)
    return f


def process_call(call, agreed, singles, raws, params, st):
    cid = call["call_id"]
    scores = {id(g): s for g, _, s in M.match_facts([a["gl"] for a in agreed], [a["ge"] for a in agreed])}
    silver, queue = [], []
    for a in agreed:
        ge, gl = a["ge"], a["gl"]
        if validate_fact(ge, call) or validate_fact(gl, call):
            st["schema_invalid"] += 1
            continue
        reasons, ta = pair_checks(ge, gl, call)
        same = "source_turn" not in reasons
        fact = canonical(ge, gl) if same else strip(ge)
        rec = {"call_id": cid, "priority": 1.0, "fact": fact, "gliner": {**side(gl), "conf": gl["conf"]}, "gemma": side(ge)}
        if not reasons:
            f = copy.deepcopy(fact)
            f["meta"] = {"label_source": "silver", "sources": ["gliner", "gemma"], "confidence": gl["conf"],
                         "match_score": scores.get(id(gl)), "temporal_agreed": ta,
                         "gliner": side(gl), "gemma": side(ge)}
            silver.append(f)
            if not ta:
                queue.append({**rec, "reasons": ["dates"]})
        else:
            queue.append({**rec, "reasons": reasons + ([] if ta else ["dates"])})
    for src, p in singles:
        if validate_fact(p, call):
            st["schema_invalid"] += 1
            continue
        r = ["gemma_only" if src == "ge" else "gliner_only"]
        if not grounded(p, call):
            r.append("ungrounded")
        pr = E.score(params, E.features(cid, src, p, agreed, singles, raws, call))
        queue.append({"call_id": cid, "reasons": r, "priority": pr, "fact": strip(p),
                      "gliner": {**side(p), "conf": p.get("conf")} if src == "gl" else None,
                      "gemma": side(p) if src == "ge" else None})
    # dedup silver
    best = {}
    for f in silver:
        k = (f["attribute"], f["value"], f["source"]["turn_idx"])
        if k not in best or f["meta"]["confidence"] > best[k]["meta"]["confidence"]:
            best[k] = f
    st["dedup_silver"] += len(silver) - len(best)
    out, seen = [], set()
    for f in silver:
        if best.get((f["attribute"], f["value"], f["source"]["turn_idx"])) is not f:
            continue
        if f["attribute"] == "agreement":
            if f["value"] in seen:
                st["dedup_silver"] += 1
                continue
            seen.add(f["value"])
        out.append(f)
    # dedup queue
    qb = {}
    for q in queue:
        k = (tuple(q["reasons"]), q["fact"]["attribute"], q["fact"]["value"], q["fact"]["source"]["turn_idx"])
        if k not in qb or q["priority"] > qb[k]["priority"]:
            qb[k] = q
    st["dedup_queue"] += len(queue) - len(qb)
    return out, [q for q in queue if qb[(tuple(q["reasons"]), q["fact"]["attribute"], q["fact"]["value"], q["fact"]["source"]["turn_idx"])] is q]


def eval_md(calls, silver_by, queue):
    groups = collections.OrderedDict()
    g = lambda k: groups.setdefault(k, collections.defaultdict(list))
    for cid, fs in silver_by.items():
        for f in fs:
            g("silver")[cid].append(f)
            g("silver temporal_agreed" if f["meta"]["temporal_agreed"] else "silver temporal disputed")[cid].append(f)
    for q in queue:
        for r in q["reasons"]:
            g("queue: " + r)[q["call_id"]].append(q["fact"])
    top = sorted(queue, key=lambda q: -q["priority"])[:max(1, len(queue) // 5)]
    for q in top:
        g("queue top 20% by priority")[q["call_id"]].append(q["fact"])
    rows = ["| группа | n | precision | valid_from верна | valid_to верна |", "|---|---|---|---|---|"]
    byid = {c["call_id"]: c for c in calls}
    for name, per in groups.items():
        n = ok = nt = vf = vt = 0
        for cid, fs in per.items():
            n += len(fs)
            for gf, pf, _ in M.match_facts(byid[cid]["facts"], fs):
                ok += 1
                if gf["attribute"] in M.TEMPORAL_ATTRS:
                    nt += 1
                    vf += _day(gf["valid_from"]) == _day(pf["valid_from"])
                    vt += _day(gf.get("valid_to")) == _day(pf["valid_to"])
        rows.append(f"| {name} | {n} | {ok / n:.3f} | {vf / nt if nt else float('nan'):.3f} | {vt / nt if nt else float('nan'):.3f} |")
    return "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--gemma", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pj = json.load(open(a.params))
    params = E.model_of(pj)
    calls, raws, gemma = E.load_inputs(a.calls, a.raw, a.gemma)
    pool = E.build_pool(calls, raws, gemma)
    st = collections.Counter()
    silver_by, queue = {}, []
    for c in calls:
        agreed, singles = pool[c["call_id"]]
        s, q = process_call(c, agreed, singles, raws, params, st)
        for f in s:
            err = validate_fact(f, c)
            if err:
                raise SystemExit(f"{c['call_id']}: invalid silver fact {err}: {f}")
        silver_by[c["call_id"]] = s
        queue += q
    queue.sort(key=lambda q: (-q["priority"], q["call_id"]))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "silver.jsonl"), "w", encoding="utf-8") as fh:
        for c in calls:
            fh.write(json.dumps({"call_id": c["call_id"], "label_source": "silver", "facts": silver_by[c["call_id"]]}, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "review_queue.jsonl"), "w", encoding="utf-8") as fh:
        for q in queue:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    qc = collections.Counter(r for q in queue for r in q["reasons"])
    stats = {"calls": len(calls), "silver_facts": sum(map(len, silver_by.values())),
             "silver_temporal_agreed": sum(f["meta"]["temporal_agreed"] for v in silver_by.values() for f in v),
             "queue": dict(qc), "schema_invalid": st["schema_invalid"], "dedup_silver": st["dedup_silver"],
             "dedup_queue": st["dedup_queue"],
             "inputs": {"calls": a.calls, "raw": a.raw, "gemma": a.gemma, "params": a.params}}
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), ensure_ascii=False, indent=1)
    if all("facts" in c for c in calls):
        open(os.path.join(a.out, "eval.md"), "w").write(eval_md(calls, silver_by, queue))


if __name__ == "__main__":
    main()
