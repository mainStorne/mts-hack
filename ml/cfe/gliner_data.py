#!/usr/bin/env python3
"""Данные GLiNER2.5 для классификационной постановки CFE. Только stdlib."""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr_noise  # noqa: E402
import clf_task as T  # noqa: E402

NAME_TYPE = "имя клиента"
TRAP_RE = re.compile(r"\?|предлож|можем|могу подключ|хотите|давайте подключ|акци|скидк")
NEG_KEEP_TRAP = 0.6
NEG_KEEP = 0.2


def build(calls, catalog, vl, rng):
    objs = set(T.object_labels(catalog))
    rows = {"det": [], "typed": [], "name": []}
    for c in calls:
        gs = T.gold_slots(c)
        for t in range(len(c["turns"])):
            attrs = [a for a in T.ATTRS if (t, a) in gs]
            if not attrs and rng.random() >= (NEG_KEEP_TRAP if TRAP_RE.search(c["turns"][t]["text"].lower()) else NEG_KEEP):
                continue
            labels = [T.ATTR_RU[a] for a in T.ATTRS] + [T.NO_FACT]
            rows["det"].append({"input": T.render_context(c, t), "output": {"classifications": [{
                "task": "attrs", "labels": labels, "multi_label": True,
                "true_label": [T.ATTR_RU[a] for a in attrs] or [T.NO_FACT]}]}})
        for (t, a), f in gs.items():
            cl = []
            if f["value"] in vl[a]:
                cl.append({"task": f"value_{a}", "labels": vl[a], "true_label": [f["value"]]})
            if a == "service_state" and f["object"]["name"] in objs:
                cl.append({"task": "object", "labels": T.object_labels(catalog), "true_label": [f["object"]["name"]]})
            if not cl:
                continue
            rows["typed"].append({"input": T.render_context(c, t) + "\nТип факта: " + T.ATTR_RU[a],
                                  "output": {"classifications": cl}})
        text = T.transcript_head(c)
        names = [f["object"]["name"].split()[0] for f in c["facts"] if f["object"]["type"] == "client" and f["object"]["name"].split()]
        ent = [names[0]] if names and re.search(r"(?<!\w)" + re.escape(names[0]) + r"(?!\w)", text) else []
        rows["name"].append({"input": text, "output": {"entities": {NAME_TYPE: ent}}})
    return rows


def main():
    cat = T.load_catalog()
    vl = T.value_labels(cat)
    calls = T.load_calls(T.TRAIN_PATH)
    rng = random.Random(0)
    ids = sorted({c["call_id"] for c in calls})
    rng.shuffle(ids)
    dev_ids = set(ids[:max(1, len(ids) // 20)])
    d = os.path.join(T.ROOT, "data")
    for name, sel in (("train", [c for c in calls if c["call_id"] not in dev_ids]),
                      ("dev", [c for c in calls if c["call_id"] in dev_ids])):
        if name == "train":
            aug = random.Random(2)
            sel = [asr_noise.corrupt_call(c, aug.uniform(0.1, 0.3), 0) if aug.random() < 0.5 else c for c in sel]
        else:
            with open(os.path.join(d, "dev_calls.jsonl"), "w", encoding="utf-8") as fh:
                for c in sel:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        rows = build(sel, cat, vl, random.Random(0))
        allr = rows["det"] + rows["typed"] + rows["name"]
        random.Random(1).shuffle(allr)
        with open(os.path.join(d, f"gliner_{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in allr:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, {k: len(v) for k, v in rows.items()}, "calls", len(sel))


if __name__ == "__main__":
    main()
