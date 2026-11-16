#!/usr/bin/env python3
"""Итоговая схема: action из BERT-детектора + ruT5, остальные атрибуты — из базового предсказания (ens-v3 / cfe_pipeline).

  python3 compose_final.py --gold raw/val.jsonl --base ../runs/ens-v3/pred_test.jsonl \
      --flags ../data/turn_gen/val_flagged.jsonl --gen ../runs/gen-t5/pred_val_flagged.jsonl --thr 0.2 --out final_pred.jsonl

--flags: реплики с вероятностью детектора p (build_action_data.py gen); --gen: выход gen_seq.py по этим репликам.
Факты action с value="нет" отбрасываются; дубли по (звонок, значение) схлопываются.
"""
import argparse, json
from collections import defaultdict

import case_metrics as cm


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def action_facts(gold, flags, gen, thr):
    p = {x["id"]: x["p"] for x in flags}
    out, seen = defaultdict(list), set()
    for r in gen:
        if p[r["id"]] < thr:
            continue
        cid, t = r["id"].rsplit("|", 1)
        t = int(t)
        turn = gold[cid]["turns"][t]
        for v in (s.strip() for s in r["pred"].split(" ; ")):
            if not v or v.lower() == "нет" or (cid, v) in seen:
                continue
            seen.add((cid, v))
            out[cid].append({"object": {"type": "client", "name": "клиент"}, "attribute": "action", "value": v, "polarity": 1,
                             "speaker": turn["speaker"], "valid_from": gold[cid]["started_at"], "valid_to": None,
                             "source": {"turn_idx": t, "span": None, "quote": None}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--flags", required=True); ap.add_argument("--gen", required=True)
    ap.add_argument("--thr", type=float, default=0.2); ap.add_argument("--out")
    a = ap.parse_args()
    gold = {c["call_id"]: c for c in load(a.gold)}
    cm.load_catalog(a.gold)
    base = {c["call_id"]: c for c in load(a.base)}
    new = action_facts(gold, load(a.flags), load(a.gen), a.thr)
    final = {cid: {"call_id": cid, "started_at": g["started_at"],
                   "facts": [f for f in base[cid]["facts"] if f["attribute"] != "action"] + new.get(cid, [])}
             for cid, g in gold.items()}
    for name, pred in (("base", base), ("final", final)):
        m = cm.evaluate(gold, pred)
        p, r = m["fact_precision"], m["fact_recall"]
        print(f"{name:6} P={p:.3f} R={r:.3f} F1={2 * p * r / (p + r):.3f} temporal={m['temporal_recall']:.3f} "
              f"state={m['state_accuracy']:.3f} grounding={m['grounding_turn']:.3f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for c in final.values():
                f.write(json.dumps(c, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
