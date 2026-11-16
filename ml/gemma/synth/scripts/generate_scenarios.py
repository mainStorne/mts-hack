#!/usr/bin/env python3
"""Генерация структурированных сценариев (этап 1: сценарий).

Каждый сценарий = семья + seed. Рендерится в чистом виде (без ASR-шума),
чтобы зафиксировать событийный план и эталонные факты до наложения шума.
Выход: data/scenarios.jsonl (сценарий + чистый звонок с метками).
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synth_lib import scenario as S  # noqa: E402
from synth_lib.builder import Builder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument("--out", default=str(ROOT / "data" / "scenarios.jsonl"))
    a = ap.parse_args()

    rng = random.Random(a.seed)
    rows, seen = [], set()
    for i in range(a.n):
        fam = rng.choices(S.FAMILIES, weights=S.WEIGHTS, k=1)[0]
        # детерминированный внутренний seed на сценарий
        sseed = rng.randrange(10 ** 9)
        b = Builder(random.Random(sseed))
        m = {"family": fam.__name__}
        fam(b, m)
        call = b.finalize(noise="clean")
        key = tuple(sorted(t["text"] for t in call["turns"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "scenario_id": f"syn-{i:05d}",
            "seed": sseed,
            "family": fam.__name__,
            "scenario_type": m.get("scenario_type"),
            "noise": None,  # выбирается на этапе диалога
            "call_clean": call,
        })
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"сценариев: {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
