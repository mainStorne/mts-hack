#!/usr/bin/env python3
"""Этап 2: наложение ASR-шума на сценарии -> финальные диалоги с разметкой.

Детерминированно: сценарий (seed) перерендеривается тем же генератором,
уровень шума выбирается смесью clean/light/moderate. span/quote пересчитываются
по финальному тексту, поэтому разметка остаётся точной.

Выход: data/synthetic_dialogues.jsonl (схема кейса + поле _meta).
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synth_lib import scenario as S  # noqa: E402
from synth_lib.builder import Builder  # noqa: E402

NOISE_MIX = [("clean", 0.25), ("light", 0.45), ("moderate", 0.30)]


def render(scen: dict, noise: str) -> dict:
    b = Builder(random.Random(scen["seed"]))
    fam = next(f for f in S.FAMILIES if f.__name__ == scen["family"])
    fam(b, {"family": scen["family"]})
    return b.finalize(noise=noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(ROOT / "data" / "scenarios.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic_dialogues.jsonl"))
    ap.add_argument("--seed", type=int, default=20260502)
    ap.add_argument("--max-chars", type=int, default=4200,
                    help="отбраковка слишком длинных промптов")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    scen_rows = [json.loads(l) for l in open(a.scenarios, encoding="utf-8")]
    out, skipped = [], Counter()
    for sc in scen_rows:
        r = rng.random()
        acc = 0.0
        noise = "moderate"
        for name, w in NOISE_MIX:
            acc += w
            if r < acc:
                noise = name
                break
        try:
            call = render(sc, noise)
        except Exception as e:
            skipped[f"render_fail:{type(e).__name__}"] += 1
            continue
        text_len = sum(len(t["text"]) for t in call["turns"]) + len(json.dumps(call["facts"], ensure_ascii=False))
        if text_len > a.max_chars:
            skipped["too_long"] += 1
            continue
        call["call_id"] = f"SYN_{sc['scenario_id']}"
        call["_meta"] = {
            "origin": "generated",
            "scenario_id": sc["scenario_id"],
            "family": sc["family"],
            "scenario_type": sc["scenario_type"],
            "seed": sc["seed"],
            "noise": noise,
            "generator": "synth_lib v1 (template scenario engine)",
        }
        out.append(call)
    with open(a.out, "w", encoding="utf-8") as fh:
        for c in out:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"диалогов: {len(out)}, отброшено: {dict(skipped)} -> {a.out}")


if __name__ == "__main__":
    main()
