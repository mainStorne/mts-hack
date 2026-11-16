#!/usr/bin/env python3
"""Сравнение распределений: синтетический пилот vs gold train (+ аугменты).

Метрики (по кейсу CompactFactExtraction):
  * размер: звонки, реплики/звонок, длина реплики, длительность;
  * факты: на звонок, по attribute, по object.type, по polarity, по speaker;
  * темпоральность: смещение valid_from относительно дня звонка (прошлое/будущее),
    доля закрытых valid_to, часовых меток (не только день);
  * source: доля quote=null, распределение turn_idx;
  * шум: noise_level.

Выход: reports/DISTRIBUTION_REPORT.md + JSON рядом.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ATTRS = ["service_state", "tariff", "ticket_status", "action", "agreement"]
OTYPES = ["client", "service", "ticket", "device"]


def _day(iso: str) -> date:
    return date.fromisoformat(iso[:10])


def collect(path: str) -> dict:
    calls = [json.loads(l) for l in open(path, encoding="utf-8")]
    n_calls = len(calls)
    turns_per = [len(c["turns"]) for c in calls]
    tlen = [len(t["text"]) for c in calls for t in c["turns"]]
    facts = [f for c in calls for f in c.get("facts", [])]
    fpk = [len(c.get("facts", [])) for c in calls]

    off = Counter()
    for c in calls:
        day = _day(c["started_at"])
        for f in c.get("facts", []):
            d = (_day(f["valid_from"]) - day).days
            off[min(max(d, -40), 14)] += 1
    past = sum(v for k, v in off.items() if k < 0)
    same = off[0]
    fut = sum(v for k, v in off.items() if k > 0)
    closed = sum(1 for f in facts if f.get("valid_to"))
    with_hour = sum(1 for f in facts if f["valid_from"][11:] != "00:00:00")
    qnull = Counter()
    per_attr = Counter(f["attribute"] for f in facts)
    for f in facts:
        if f["source"].get("quote") is None:
            qnull[f["attribute"]] += 1
    ti = Counter(f["source"]["turn_idx"] for f in facts)
    spk = Counter(f["speaker"] for f in facts)
    noise = Counter(c.get("noise_level") for c in calls)
    return {
        "file": path,
        "calls": n_calls,
        "turns_per_call": round(sum(turns_per) / n_calls, 1),
        "turn_chars": round(sum(tlen) / len(tlen), 0),
        "facts_total": len(facts),
        "facts_per_call": round(sum(fpk) / n_calls, 2),
        "by_attribute": {a: per_attr.get(a, 0) for a in ATTRS},
        "by_attribute_share": {a: round(per_attr.get(a, 0) / max(1, len(facts)), 3)
                               for a in ATTRS},
        "by_otype": {o: sum(1 for f in facts if f["object"]["type"] == o) for o in OTYPES},
        "polarity_-1": sum(1 for f in facts if f["polarity"] == -1),
        "speaker_client_share": round(spk.get("client", 0) / max(1, len(facts)), 3),
        "vf_past_share": round(past / max(1, len(facts)), 3),
        "vf_same_day_share": round(same / max(1, len(facts)), 3),
        "vf_future_share": round(fut / max(1, len(facts)), 3),
        "vf_offsets_top": off.most_common(12),
        "valid_to_closed_share": round(closed / max(1, len(facts)), 3),
        "with_hour_share": round(with_hour / max(1, len(facts)), 3),
        "quote_null_share": {a: round(qnull.get(a, 0) / max(1, per_attr.get(a, 1)), 3)
                             for a in ATTRS},
        "turn_idx_top": ti.most_common(10),
        "noise_levels": {str(k): v for k, v in noise.items()},
    }


def fmt_row(name: str, keys: list[str], a: dict, b: dict) -> str:
    cells = []
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict):
            cells.append(f"{va.get('service_state')}/{va.get('tariff')}/"
                         f"{va.get('ticket_status')}/{va.get('action')}/{va.get('agreement')}")
        else:
            cells.append(f"{va} / {vb}")
    return f"| {name} | " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default=str(ROOT / "data/synthetic_pilot.jsonl"))
    ap.add_argument("--train", default=str(ROOT.parent / "data/raw/train.jsonl"))
    ap.add_argument("--augmented", default=str(ROOT / "data/augmented_train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "reports/DISTRIBUTION_REPORT.md"))
    a = ap.parse_args()

    syn = collect(a.synthetic)
    tr = collect(a.train)
    aug = collect(a.augmented) if a.augmented and Path(a.augmented).exists() else None

    keys = ["calls", "turns_per_call", "turn_chars", "facts_total", "facts_per_call",
            "by_attribute", "by_otype", "polarity_-1", "speaker_client_share",
            "vf_past_share", "vf_same_day_share", "vf_future_share",
            "valid_to_closed_share", "with_hour_share", "quote_null_share"]
    lines = [
        "# Распределение синтетического пилота vs gold train",
        "",
        "Синтетика: template scenario engine (подход A), 27 семей сценариев.",
        "",
        "| метрика | синтетика | train" + (" | аугменты (подход B) |" if aug else " |"),
        "|---" * (3 + (1 if aug else 0)) + "|",
    ]
    for k in keys:
        row = f"| {k} | {syn.get(k)} | {tr.get(k)} |"
        if aug:
            row += f" {aug.get(k)} |"
        lines.append(row)
    lines += [
        "",
        "## Смещения valid_from относительно дня звонка (синтетика)",
        "",
        "`" + json.dumps(syn["vf_offsets_top"], ensure_ascii=False) + "`",
        "",
        "## source.turn_idx (синтетика)",
        "",
        "`" + json.dumps(syn["turn_idx_top"], ensure_ascii=False) + "`",
        "",
        "## Шум (синтетика)",
        "",
        "`" + json.dumps(syn["noise_levels"], ensure_ascii=False) + "`",
        "",
        "## Выводы",
        "",
        f"- Смешение attribute и объектных типов в синтетике повторяет пропорции train "
        f"(доминируют service_state {syn['by_attribute_share']['service_state']:.0%} и "
        f"ticket_status {syn['by_attribute_share']['ticket_status']:.0%}; в train — "
        f"{tr['by_attribute_share']['service_state']:.0%} и "
        f"{tr['by_attribute_share']['ticket_status']:.0%}).",
        "- Темпоральная структура: в синтетике представлены прошлые состояния (цепочки), "
        "будущие даты (согласования/заявки) и день звонка — как в train.",
        f"- Расхождение: в синтетике больше фактов с датой в прошлом "
        f"({syn['vf_past_share']:.0%} против {tr['vf_past_share']:.0%} в train) и меньше "
        f"«same-day» ({syn['vf_same_day_share']:.0%} против {tr['vf_same_day_share']:.0%}): "
        "жалобы в сценариях почти всегда датированы явно. Частично компенсируется реальными "
        "данными в смешанных вариантах; при масштабировании добавить undated-жалобы "
        "(vf = день звонка) в семьи service_state.",
        f"- Синтетика короче train: {syn['turns_per_call']} реплик/звонок против "
        f"{tr['turns_per_call']} и {syn['turn_chars']:.0f} символов против "
        f"{tr['turn_chars']:.0f}. Для пилота допустимо; при масштабировании — длинные "
        "семейства с несколькими темами и recap-репликами.",
        f"- Доля quote=null ниже, чем в gold "
        f"(service_state {syn['quote_null_share']['service_state']:.0%} против "
        f"{tr['quote_null_share']['service_state']:.0%}): синтетика сознательно более "
        "«заземлена», каждый факт привязан к цитате либо к реплике-носителю.",
    ]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"отчёт -> {out}")
    print(json.dumps({"synthetic": syn, "train": tr}, ensure_ascii=False)[:400] + " …")


if __name__ == "__main__":
    main()
