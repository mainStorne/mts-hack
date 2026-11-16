"""Доля верного дня valid_from на gold-фактах: старое правило, prior и политика. Только stdlib (+pymorphy3 при --morph)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec, rules  # noqa: E402

def acc(calls, fn):
    F = [(c, f) for c in calls for f in c["facts"]]
    return round(sum(fn(c, f) == codec.day_offset(c["started_at"], f["valid_from"]) for c, f in F) / len(F), 4)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph", action="store_true")
    ap.add_argument("--sentence", action="store_true")
    a = ap.parse_args()
    rules.MORPH, rules.SAME_SENTENCE = a.morph, a.sentence
    L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
    dm = rules.build_date_model(L("data/raw/train.jsonl"))
    out = {"morph": a.morph, "sentence": a.sentence}
    for name, p in (("val", "data/raw/val.jsonl"), ("val_noisy", "data/val_noisy.jsonl")):
        C = L(p)
        out[name] = {"rules": acc(C, lambda c, f: rules.pick_mention(c, f["source"]["turn_idx"], f["value"], f["attribute"])[0]),
                     "policy": acc(C, lambda c, f: rules.resolve_date(dm, c, f["source"]["turn_idx"], f["value"], f["attribute"]))}
    print(json.dumps(out, ensure_ascii=False))
