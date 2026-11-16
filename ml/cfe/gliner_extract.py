#!/usr/bin/env python3
"""Инференс GLiNER2.5: детекция и типизация моделью, дата и полярность в rules.py; пишет answers.jsonl и отчёт."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clf_eval  # noqa: E402
import clf_task as T  # noqa: E402


def main():
    from pipeline import Pipeline
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default=os.path.join(T.ROOT, "data/raw/val.jsonl"))
    ap.add_argument("--model", default="runs/gliner-v2/best")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--det-min", type=float, default=0.3)
    a = ap.parse_args()
    pipe = Pipeline(a.model, a.det_min)
    cat = pipe.catalog
    calls = T.load_calls(a.calls)[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "answers.jsonl")
    done = clf_eval.load_answers(path) if os.path.exists(path) else {}
    fh = open(path, "a", encoding="utf-8")
    for n, c in enumerate(calls):
        if c["call_id"] in done:
            continue
        fh.write(json.dumps(pipe.answers(c), ensure_ascii=False) + "\n")
        fh.flush()
        print(f"{n + 1}/{len(calls)}", flush=True)
    fh.close()
    clf_eval.report(calls, clf_eval.load_answers(path), a.out, "gliner", cat)


if __name__ == "__main__":
    main()
