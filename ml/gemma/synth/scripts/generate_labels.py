#!/usr/bin/env python3
"""Этап 3: сборка финального пилотного датасета и эталонной разметки.

Вход : data/synthetic_dialogues.jsonl (после validate_synthetic.py, опционально
       с --validated-ids для отбора только прошедших проверки).
Выход: data/synthetic_pilot.jsonl      — звонки строго по официальной схеме;
       data/synthetic_metadata.jsonl   — provenance в отдельном файле;
       data/labels_table.jsonl         — плоская таблица фактов для QC-отчётов.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_FIELDS = ["call_id", "started_at", "duration_sec", "series", "turns",
                 "noise_level", "facts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", default=str(ROOT / "data" / "synthetic_dialogues.jsonl"))
    ap.add_argument("--validated-ids", default=None,
                    help="файл со списком call_id, прошедших валидацию (по одному в строке)")
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    rows = [json.loads(l) for l in open(a.dialogues, encoding="utf-8")]
    allowed = None
    if a.validated_ids:
        allowed = {l.strip() for l in open(a.validated_ids, encoding="utf-8") if l.strip()}
        rows = [r for r in rows if r["call_id"] in allowed]
        print(f"отобрано прошедших валидацию: {len(rows)} из {len(allowed)}")

    n_facts = 0
    with open(out_dir / "synthetic_pilot.jsonl", "w", encoding="utf-8") as fh, \
         open(out_dir / "synthetic_metadata.jsonl", "w", encoding="utf-8") as fm, \
         open(out_dir / "labels_table.jsonl", "w", encoding="utf-8") as fl:
        for r in rows:
            meta = r.pop("_meta", {})
            public = {k: r[k] for k in PUBLIC_FIELDS if k in r}
            # поле series обязано присутствовать (может быть null)
            public.setdefault("series", None)
            fh.write(json.dumps(public, ensure_ascii=False) + "\n")
            fm.write(json.dumps({"call_id": r["call_id"], **meta}, ensure_ascii=False) + "\n")
            for f in public["facts"]:
                n_facts += 1
                fl.write(json.dumps({
                    "call_id": r["call_id"],
                    "object_type": f["object"]["type"],
                    "object_name": f["object"]["name"],
                    "attribute": f["attribute"],
                    "value": f["value"],
                    "polarity": f["polarity"],
                    "speaker": f["speaker"],
                    "valid_from": f["valid_from"],
                    "valid_to": f["valid_to"],
                    "turn_idx": f["source"]["turn_idx"],
                    "has_span": f["source"].get("span") is not None,
                    "has_quote": f["source"].get("quote") is not None,
                    "quote": f["source"].get("quote"),
                }, ensure_ascii=False) + "\n")
    print(f"звонков: {len(rows)}, фактов: {n_facts} -> {out_dir}/synthetic_pilot.jsonl")


if __name__ == "__main__":
    main()
