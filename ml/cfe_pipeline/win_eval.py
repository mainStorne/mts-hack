#!/usr/bin/env python3
"""Оценка оконной схемы (BERT-детектор с метками окна + ruT5) для service_state / tariff на val.

Порог детектора выбирается перекрёстно по звонкам val. state считается без утечки: при равных valid_from берём
более позднюю реплику (метрика кейса при ничьей берёт значение, ближайшее к gold).

  python3 win_eval.py --attrs service_state,tariff
"""
import argparse, collections, json, pickle, random, sys

import numpy as np

import case_metrics as cm

GRID = (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)
ld = lambda p: {c["call_id"]: c for c in map(json.loads, open(p, encoding="utf-8"))}


def need(a, b):
    return cm.VALUE_SIM if min(len(cm._tokens(a)), len(cm._tokens(b))) <= 2 else cm.VALUE_SIM_LONG


def state_acc(gold, pred, attrs):
    tot = ok = 0
    for cid, g in gold.items():
        pf = pred[cid]["facts"]
        by = {}
        for f in g["facts"]:
            if f["attribute"] in attrs and f.get("valid_to") is None:
                k = (f["object"]["id"], f["attribute"])
                if k not in by or f["valid_from"] > by[k]["valid_from"]:
                    by[k] = f
        for (_, a), gf in by.items():
            tot += 1
            cs = [p for p in pf if p.get("attribute") == a and p.get("valid_to") is None and cm._object_sim(gf, p) > 0]
            if cs:
                last = max(cs, key=lambda p: (p.get("valid_from") or "", p["source"].get("turn_idx") or 0))
                ok += cm._value_sim(gf["value"], last["value"]) >= need(gf["value"], last["value"])
    return ok / max(1, tot), tot


def prf(gold, pred, attr, calls=None):
    calls = calls or set(gold)
    sel = {k: {**gold[k], "facts": [f for f in gold[k]["facts"] if f["attribute"] == attr]} for k in calls}
    pr = {k: {"call_id": k, "facts": [f for f in pred[k]["facts"] if f["attribute"] == attr]} for k in calls}
    m = cm.evaluate(sel, pr)
    p, r = m["fact_precision"], m["fact_recall"]
    return p, r, 2 * p * r / max(p + r, 1e-9)


def new_facts(attr, tau, calls, gen, p, gold, base):
    out = collections.defaultdict(list)
    for k, s in gen.items():
        cid, t = k.rsplit("|", 1)
        t = int(t)
        if cid not in calls or p[k] < tau:
            continue
        for part in s.split(" ; "):
            part = part.strip()
            if not part or part.lower() == "нет":
                continue
            name, val = (part.split(" — ", 1) if attr == "service_state" and " — " in part else ("", part))
            name, val = name.strip(), val.strip()
            if attr != "service_state":
                name = "клиент"
            if any(cm._value_sim(x["value"], val) >= 0.85 and cm._value_sim(x["object"]["name"], name) >= 0.85
                   for x in out[cid]):
                continue
            vf, vt = gold[cid]["started_at"], None
            for b in base[cid]["facts"]:
                if b["attribute"] == attr and abs((b["source"]["turn_idx"] or 0) - t) <= 1:
                    vf, vt = b["valid_from"], b["valid_to"]
                    break
            out[cid].append({"object": {"type": "client" if attr == "tariff" else "service", "name": name}, "attribute": attr,
                             "value": val, "polarity": 1, "speaker": gold[cid]["turns"][t]["speaker"], "valid_from": vf,
                             "valid_to": vt, "source": {"turn_idx": t, "span": None, "quote": None}})
    return out


def replace(base, attr, new, calls, merge=False):
    out = {}
    for cid, c in base.items():
        keep = [f for f in c["facts"] if f["attribute"] != attr]
        old = [f for f in c["facts"] if f["attribute"] == attr]
        nf = new.get(cid, []) if cid in calls else old
        if merge and cid in calls:
            used = {id(x) for x, _, _ in cm.match_facts(nf, old)} if nf and old else set()
            nf = old + [f for f in nf if id(f) not in used]
        out[cid] = {**c, "facts": keep + nf}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", default="service_state,tariff")
    a = ap.parse_args()
    gold = ld("raw/val.jsonl")
    bases = {"GLiNER v21": ld("../runs/struct-lora-v21/pred_test.jsonl"), "cfe_pipeline": ld("pred_val.jsonl"),
             "ens-v3": ld("../runs/ens-v3/pred_test.jsonl")}
    ids = sorted(gold)
    random.Random(0).shuffle(ids)
    folds = [set(ids[::2]), set(ids[1::2])]
    for attr in a.attrs.split(","):
        vr = [json.loads(l) for l in open(f"../data/turn_win_{attr}/val_clean.jsonl")]
        pb = np.load(f"../runs/turn-bert-win-{attr}/pred_val_ep0.npy")
        p = {f"{r['cid']}|{r['turn']}": float(x) for r, x in zip(vr, pb)}
        gen = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(f"../runs/gen-t5-win-{attr}/pred_gen_val_all.jsonl")}
        print(f"\n== {attr}")
        for n, b in bases.items():
            print(f"  {n:13} P/R/F1 = {tuple(round(x, 3) for x in prf(gold, b, attr))}  state(no-leak) = {state_acc(gold, b, {attr})[0]:.3f}")

        def cross(base, merge):
            res, taus = {}, []
            for k in (0, 1):
                tune, test = folds[1 - k], folds[k]
                f = lambda t: prf(gold, replace(base, attr, new_facts(attr, t, tune, gen, p, gold, base), tune, merge), attr, tune)[2]
                tau = max(GRID, key=f)
                taus.append(tau)
                res.update(new_facts(attr, tau, test, gen, p, gold, base))
            return replace(base, attr, res, set(gold), merge), taus

        for name, base, merge in (("window BERT+ruT5 alone", bases["ens-v3"], False),
                                  ("  ∪ GLiNER v21", bases["GLiNER v21"], True), ("  ∪ cfe_pipeline", bases["cfe_pipeline"], True),
                                  ("  ∪ ens-v3", bases["ens-v3"], True)):
            pred, taus = cross(base, merge)
            print(f"  {name:24} P/R/F1 = {tuple(round(x, 3) for x in prf(gold, pred, attr))}  state(no-leak) = "
                  f"{state_acc(gold, pred, {attr})[0]:.3f}  tau={taus}")


if __name__ == "__main__":
    main()
