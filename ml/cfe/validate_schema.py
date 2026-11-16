#!/usr/bin/env python3
"""Проверка выходной схемы (data/raw/schema.md §2) на всех звонках. Только stdlib.

python3 cfe/validate_schema.py CALLS.jsonl PRED.jsonl -> `calls N valid_calls N facts N violations N`, код 1 при нарушениях.
"""
import json
import sys
from datetime import datetime

KEYS = {"object", "attribute", "value", "polarity", "speaker", "valid_from", "valid_to", "source"}
OBJ_TYPES = {"service", "device", "ticket", "client", "other"}
ATTRS = {"service_state", "tariff", "ticket_status", "action", "agreement"}


def _iso(v):
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00") if isinstance(v, str) else v)
    except (TypeError, ValueError):
        return None


def _cmp(a, b):
    """a < b с учётом смеси aware/naive."""
    if (a.tzinfo is None) != (b.tzinfo is None):
        a, b = a.replace(tzinfo=None), b.replace(tzinfo=None)
    return a < b


def _nes(v):
    return isinstance(v, str) and v.strip() != ""


def check_fact(f, turns):
    if not isinstance(f, dict):
        return ["факт не объект"]
    errs = []
    if set(f) - {"confirmed"} != KEYS:  # confirmed — служебное поле gold-фактов (schema.md §2)
        return [f"ключи {sorted(f)}"]
    o = f["object"]
    if not isinstance(o, dict) or o.get("type") not in OBJ_TYPES or not isinstance(o.get("name"), str):  # имя может быть пустым (schema.md §3)
        errs.append("object")
    if f["attribute"] not in ATTRS:
        errs.append("attribute")
    if not _nes(f["value"]):
        errs.append("value")
    if isinstance(f["polarity"], bool) or f["polarity"] not in (1, -1):
        errs.append("polarity")
    if f["speaker"] not in ("client", "operator"):
        errs.append("speaker")
    vf = _iso(f["valid_from"])
    if vf is None:
        errs.append("valid_from")
    if f["valid_to"] is not None:
        vt = _iso(f["valid_to"])
        if vt is None:
            errs.append("valid_to")
        elif vf is not None and _cmp(vt, vf):
            errs.append("valid_to < valid_from")
    s = f["source"]
    if not isinstance(s, dict) or set(s) != {"turn_idx", "span", "quote"}:
        return errs + ["source ключи"]
    ti = s["turn_idx"]
    if isinstance(ti, bool) or not isinstance(ti, int) or ti not in turns:
        errs.append("source.turn_idx")
    else:
        if f["speaker"] != turns[ti]["speaker"]:
            errs.append("speaker != реплика")
        sp = s["span"]
        if sp is not None:
            if not (isinstance(sp, list) and len(sp) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in sp)
                    and 0 <= sp[0] < sp[1] <= len(turns[ti]["text"])):
                errs.append("source.span")
    if s["quote"] is not None and not isinstance(s["quote"], str):
        errs.append("source.quote")
    return errs


def validate(calls, preds):
    """-> список нарушений `call_id#i: причина` (i — номер факта, `-` — уровень звонка)."""
    out = []
    pm = {}
    for p in preds:
        cid = p.get("call_id")
        if cid in pm:
            out.append(f"{cid}#-: дубликат call_id")
        pm[cid] = p
    ids = {c["call_id"] for c in calls}
    for cid in pm:
        if cid not in ids:
            out.append(f"{cid}#-: лишний call_id")
    for c in calls:
        cid = c["call_id"]
        p = pm.get(cid)
        if p is None:
            out.append(f"{cid}#-: нет предсказания")
            continue
        if not isinstance(p.get("facts"), list):
            out.append(f"{cid}#-: facts не список")
            continue
        turns = {(t["idx"] if t.get("idx") is not None else i): t for i, t in enumerate(c["turns"])}
        for i, f in enumerate(p["facts"]):
            out.extend(f"{cid}#{i}: {e}" for e in check_fact(f, turns))
    return out


def stats(calls, preds):
    v = validate(calls, preds)
    bad = {x.split("#")[0] for x in v}
    nf = sum(len(p["facts"]) for p in preds if isinstance(p.get("facts"), list))
    return {"calls": len(calls), "invalid_calls": len(bad), "facts": nf, "violations": len(v), "list": v}


def _read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    calls, preds = _read(sys.argv[1]), _read(sys.argv[2])
    r = stats(calls, preds)
    print(f"calls {r['calls']} valid_calls {r['calls'] - r['invalid_calls']} facts {r['facts']} violations {r['violations']}")
    for x in r["list"][:20]:
        print(x)
    sys.exit(1 if r["violations"] else 0)
