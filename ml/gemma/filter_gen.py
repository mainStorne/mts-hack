"""Drop generated calls whose facts point at the wrong turn.

A call is dropped whole if any fact has: turn_idx out of range; speaker differing
from the cited turn's speaker; or a non-empty quote absent from the cited turn
but present in another turn.
"""
import argparse
import json


def bad_fact(fact, turns):
    ti = fact["source"]["turn_idx"]
    if not isinstance(ti, int) or not 0 <= ti < len(turns):
        return True
    if fact.get("speaker") != turns[ti]["speaker"]:
        return True
    q = (fact["source"].get("quote") or "").lower()
    if q and q not in turns[ti]["text"].lower():
        return any(q in t["text"].lower() for j, t in enumerate(turns) if j != ti)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kept = dropped = 0
    with open(a.inp, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as o:
        for line in f:
            if not line.strip():
                continue
            call = json.loads(line)
            if any(bad_fact(x, call["turns"]) for x in call["facts"]):
                dropped += 1
            else:
                kept += 1
                o.write(line if line.endswith("\n") else line + "\n")
    print(f"kept={kept} dropped={dropped}")


if __name__ == "__main__":
    main()
