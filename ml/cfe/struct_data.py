#!/usr/bin/env python3
"""Разбиение train на train/dev и обучающие записи GLiNER2 (структура object + polarity + person). Только stdlib."""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clf_task as T  # noqa: E402
import codec  # noqa: E402
import denoise as D  # noqa: E402
import struct_schema as S  # noqa: E402



def find_ci(ctx, sub):
    """Подстрока ctx, совпавшая с sub без учёта регистра, иначе ''."""
    if not sub:
        return ""
    m = re.search(r"(?<!\w)" + re.escape(sub) + r"(?!\w)", ctx, re.I)
    return m.group(0) if m else ""


def split_calls(calls):
    groups = defaultdict(list)
    for c in calls:
        key = c["series"]["account_id"] if c.get("series") else c["call_id"]
        groups[key].append(c)
    keys = sorted(groups)
    random.Random(0).shuffle(keys)
    n_dev = round(len(keys) * 0.1)
    dev = [c for k in keys[:n_dev] for c in groups[k]]
    train = [c for k in keys[n_dev:] for c in groups[k]]
    return train, dev


def records(calls, denoise=False, schema="entities"):
    pos, neg = [], []
    skipped_facts = skipped_turns = with_date = denial_spans = obj_spans = 0
    fact = schema == "fact"
    for call in calls:
        if denoise:
            call = D.denoise_call(call)[0]
        by_t = defaultdict(list)
        for f in call["facts"]:
            by_t[f["source"]["turn_idx"]].append(f)
        cname = next((f["object"]["name"] for f in call["facts"] if f["object"]["type"] == "client"), "")
        for t, turn in enumerate(call["turns"]):
            ctx = T.render_context(call, t)
            off = S.turn_offset(ctx, call, t)
            text = turn["text"]
            fs = by_t.get(t, [])
            ents = {"value": [], **{S.qlabel(g, l): [] for g, ls in S.GROUPS.items() for l in ls}, "date": []}
            structs, seen_f = [], set()

            def add(key, sp):
                if all((x["start"], x["end"]) != (sp["start"], sp["end"]) for x in ents[key]):
                    ents[key].append(sp)

            def shifted(s, e):
                return {"text": ctx[off + s:off + e], "start": off + s, "end": off + e}

            located = 0
            for f in fs:
                is_act = f["attribute"] == "action"
                query = f.get("evidence") or (f["value"].split(" — ")[0] if is_act else f["value"])
                sc, s, e = S.locate_value(text, query)
                if sc < S.TRAIN_MIN_SIM:
                    skipped_facts += 1
                    continue
                located += 1
                sp = shifted(s, e)
                pol = "denial" if f["polarity"] == -1 else "approval"
                denial_spans += pol == "denial"
                if not fact:
                    add("value", sp)
                    add(S.qlabel("attribute", f["attribute"]), sp)
                    add(S.qlabel("type", f["object"]["type"]), sp)
                    add(S.qlabel("polarity", pol), sp)
                    continue
                if (sp["start"], sp["end"], f["attribute"]) in seen_f:
                    continue
                seen_f.add((sp["start"], sp["end"], f["attribute"]))
                obj = ""
                if f["object"]["type"] in ("service", "device", "ticket"):
                    osc, os_, oe = S.locate_value(text, f["object"]["name"])
                    if osc >= S.TRAIN_MIN_SIM:
                        obj = shifted(os_, oe)
                dt = ""
                want = codec.day_offset(call["started_at"], f["valid_from"])
                for fut in (False, True):
                    for o, s2, e2, k in S.date_spans(text, call["started_at"], fut):
                        if o == want and dt == "":
                            dt = shifted(s2, e2)
                structs.append({"fact": {
                    "value": sp,
                    "attribute": {"value": f["attribute"], "choices": S.ATTRS},
                    "type": {"value": f["object"]["type"], "choices": S.TYPES},
                    "polarity": {"value": pol, "choices": S.POL},
                    "object": obj, "date": dt}})
            if fs and not located:
                skipped_turns += 1
                continue
            pn = find_ci(ctx, cname)
            if fact:
                if not structs:
                    structs = [{"fact": {k: "" for k in S.FACT_FIELDS}}]
                with_date += sum(1 for x in structs if x["fact"]["date"] != "")
                obj_spans += sum(1 for x in structs if x["fact"]["object"] != "")
                rec = {"input": ctx, "output": {"json_structures": structs,
                                                "entities": {"person": [pn] if pn else []}}}
            else:
                seen = set()
                for f in fs:
                    want = codec.day_offset(call["started_at"], f["valid_from"])
                    for fut in (False, True):
                        for o, s, e, k in S.date_spans(text, call["started_at"], fut):
                            if o == want and (s, e) not in seen:
                                seen.add((s, e))
                                ents["date"].append(shifted(s, e))
                if pn:
                    ents["person"] = [pn]
                if ents["date"]:
                    with_date += 1
                rec = {"input": ctx, "output": {"entities": ents}}
            (pos if fs else neg).append(rec)
    return pos, neg, skipped_facts, skipped_turns, with_date, denial_spans, obj_spans


def write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_gen(train, gen_dir):
    """Evidence из repair.jsonl вносится в факты train; затем дописываются сгенерированные звонки."""
    rep = {}
    p = os.path.join(gen_dir, "repair.jsonl")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            rep[(r["call_id"], r["fact_idx"])] = r["evidence"]
    n = 0
    for c in train:
        for i, f in enumerate(c["facts"]):
            ev = rep.get((c["call_id"], i))
            if ev:
                f["evidence"] = ev
                n += 1
    new = []
    p = os.path.join(gen_dir, "new_calls.jsonl")
    if os.path.exists(p):
        new = [json.loads(l) for l in open(p, encoding="utf-8")]
    print(f"gen: repaired_facts={n} new_calls={len(new)}")
    return train + new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--gen", help="каталог data/gen с repair.jsonl и new_calls.jsonl")
    ap.add_argument("--out", default="struct", help="подкаталог data/")
    ap.add_argument("--schema", choices=["entities", "fact"], default="entities")
    a = ap.parse_args()
    out = os.path.join(T.ROOT, "data", a.out)
    os.makedirs(out, exist_ok=True)
    calls = T.load_calls(T.TRAIN_PATH)
    train, dev = split_calls(calls)
    write(os.path.join(out, "train_calls.jsonl"), train)
    write(os.path.join(out, "dev_calls.jsonl"), dev)
    if a.gen:
        train = load_gen(train, a.gen)
    for name, cs in (("train", train), ("dev", dev)):
        pos, neg, sf, st, wd, dn, osp = records(cs, a.denoise, a.schema)
        neg = random.Random(0).sample(neg, min(len(pos), len(neg)))
        rows = pos + neg
        random.Random(1).shuffle(rows)
        write(os.path.join(out, f"{name}.jsonl"), rows)
        extra = f" obj_spans={osp}" if a.schema == "fact" else ""
        print(f"{name}: calls={len(cs)} pos={len(pos)} neg={len(neg)} skipped_facts={sf} skipped_turns={st} with_date(all)={wd} denial_spans={dn}{extra}")


if __name__ == "__main__":
    main()
