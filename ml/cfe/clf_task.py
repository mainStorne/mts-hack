"""Постановка CFE как классификации: общие константы и сборка фактов. Только stdlib."""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH = os.path.join(ROOT, "data", "clf_labels.json")
BASELINE_PATH = os.path.join(ROOT, "data", "raw", "baseline_metrics.json")
TRAIN_PATH = os.path.join(ROOT, "data", "raw", "train.jsonl")
CATALOG_PATH = os.path.join(ROOT, "data", "raw", "catalog.json")

ATTRS = ["service_state", "tariff", "ticket_status", "action", "agreement"]
ATTR_RU = {
    "service_state": "состояние услуги или устройства",
    "tariff": "тарифный план клиента",
    "ticket_status": "статус заявки",
    "action": "действие, которое клиент уже выполнил, и его результат",
    "agreement": "договорённость, достигнутая в звонке (выезд, возврат, перезвон…)",
}
NO_FACT = "нет факта"
EXTRA_PER_ATTR = 50
EXTRA_ATTRS = ("tariff", "agreement")  # у остальных добавки из train — запоминание конкретных звонков
FAILED_SUFFIX = " — не помогло"



def load_calls(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_catalog(path=CATALOG_PATH):
    return json.load(open(path, encoding="utf-8"))


def build_value_labels(train_path, catalog):
    """Метки значений: каталог; у action ещё «X — не помогло» для каждого X; у EXTRA_ATTRS плюс EXTRA_PER_ATTR частых строк train."""
    freq = {a: Counter() for a in ATTRS}
    for c in load_calls(train_path):
        for f in c["facts"]:
            freq[f["attribute"]][f["value"]] += 1
    out = {}
    for a in ATTRS:
        labels = list(dict.fromkeys(catalog[codec.CATALOG_CATEGORY[a]]))
        if a == "action":
            labels += [v + FAILED_SUFFIX for v in labels]
        if a not in EXTRA_ATTRS:
            out[a] = labels
            continue
        seen = set(labels)
        extra = 0
        for v, _ in freq[a].most_common():
            if extra >= EXTRA_PER_ATTR:
                break
            if v not in seen:
                labels.append(v)
                seen.add(v)
                extra += 1
        out[a] = labels
    json.dump(out, open(LABELS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def value_labels(catalog):
    if os.path.exists(LABELS_PATH):
        return json.load(open(LABELS_PATH, encoding="utf-8"))
    return build_value_labels(TRAIN_PATH, catalog)


def object_labels(catalog):
    return catalog["services"] + catalog["devices"]


def transcript_lines(call):
    return [f"[{t['idx']}] {'К' if t['speaker'] == 'client' else 'О'}: {t['text']}" for t in call["turns"]]


NAME_WINDOW = 2500


def transcript_head(call, limit=NAME_WINDOW):
    return "\n".join(transcript_lines(call))[:limit]


def render_context(call, t, radius=3):
    lines = [codec.date_header(call["started_at"])]
    turns = call["turns"]
    for i in range(max(0, t - radius), min(len(turns), t + radius + 1)):
        s = f"[{i}] {'К' if turns[i]['speaker'] == 'client' else 'О'}: {turns[i]['text']}"
        lines.append((">>> " + s) if i == t else s)
    return "\n".join(lines)


def name_candidates(call):
    seen = []
    for t in call["turns"]:
        if t["speaker"] != "client":
            continue
        for w in re.findall(r"\b[А-ЯЁ][а-яё]{2,}\b", t["text"]):
            if w not in seen:
                seen.append(w)
    return seen


def ticket_name(call, t):
    for i in range(t, -1, -1):
        m = re.search(r"\b\d{4}\b", call["turns"][i]["text"])
        if m:
            return f"заявка {m.group(0)}"
    return "заявка"


def assemble(call, slots, name, catalog):
    devices = set(catalog["devices"])
    s = call["started_at"]
    facts = []
    for sl in slots:
        t, a = sl["t"], sl["attr"]
        if a == "service_state":
            obj = sl.get("object") or ""
            o = {"type": "device" if obj in devices else "service", "name": obj}
        elif a == "ticket_status":
            o = {"type": "ticket", "name": ticket_name(call, t)}
        else:
            o = {"type": "client", "name": name or ""}
        facts.append({
            "object": o, "attribute": a, "value": sl["value"],
            "polarity": sl.get("polarity", 1),
            "speaker": call["turns"][t]["speaker"],
            "valid_from": codec.iso(s, sl["offset"]),
            "valid_to": None,
            "source": {"turn_idx": t, "span": None, "quote": None},
        })
    return codec.close_states(codec.drop_repeated_states(facts))


def gold_slots(call):
    out = {}
    for f in call["facts"]:
        out.setdefault((f["source"]["turn_idx"], f["attribute"]), f)
    return out


if __name__ == "__main__":
    cat = load_catalog()
    print({a: len(v) for a, v in build_value_labels(TRAIN_PATH, cat).items()})


def baseline():
    m = json.load(open(BASELINE_PATH, encoding="utf-8"))
    p, r = m["fact_precision"], m["fact_recall"]
    m["f1"] = round(2 * p * r / (p + r), 4)
    return m
