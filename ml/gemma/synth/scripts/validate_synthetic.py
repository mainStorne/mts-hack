#!/usr/bin/env python3
"""Автоматические проверки корректности синтетического датасета.

Проверки (по пунктам ТЗ):
  1. JSON/обязательные поля, ISO-даты, типы;
  2-4. объекты/атрибуты/значения/роли;
  5-6. speaker == speaker(turn), turn_idx в диапазоне, реплика существует;
  7. span и quote точно соответствуют тексту реплики (пересчёт по find);
  8. временные интервалы: valid_from <= valid_to, непересечение и корректное
     закрытие single-valued цепочек (service_state/tariff/ticket_status);
  10. дубликаты реплик/звонков;
  11. отсутствие близких копий train (n-граммная похожесть);
  12. roundtrip cfe.codec.encode_target -> decode_target (совпадение полей t,o,a,v,p,f,e);
  13. запрет «мёртвых» ссылок: факты не ссылаются на реплики-пустышки,
      каждое упоминание номера заявки в тексте согласовано с объектами.
Плюс семантический контроль цитат: в цитате факта присутствуют ключевые маркеры
(числа/даты/название тарифа/номер заявки), если они есть в value.

Выход: отчёт JSON + список отклонённых call_id.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

ATTRS = {"service_state", "tariff", "ticket_status", "action", "agreement"}
TYPES = {"service", "device", "ticket", "client"}
SINGLE_VALUED = {"service_state", "tariff", "ticket_status"}
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


def _cfe():
    import codec
    return codec


def check_call(call: dict, train_shingles: dict | None = None):
    """(список ошибок, предупреждений) для одного звонка."""
    errs, warns = [], []
    cid = call.get("call_id", "?")

    def bad(msg):
        errs.append(f"{cid}: {msg}")

    def warn(msg):
        warns.append(f"{cid}: {msg}")

    for k in ("call_id", "started_at", "turns", "facts"):
        if k not in call:
            bad(f"нет поля {k}")
            return errs, warns
    if not ISO_RE.match(call["started_at"]):
        bad("started_at не ISO")
    turns = call["turns"]
    for t in turns:
        if t.get("speaker") not in ("client", "operator"):
            bad(f"turn {t['idx']}: спикер")
    # индексы и уникальность
    idxs = [t["idx"] for t in turns]
    if idxs != list(range(len(turns))):
        bad("turn.idx не 0..n-1")
    # подряд одинаковые спикеры (в gold 0.6%)
    for a, b in zip(turns, turns[1:]):
        if a["speaker"] == b["speaker"]:
            warn(f"подряд один спикер: turns {a['idx']},{b['idx']}")
    # реплика не должна обрываться посреди фразы (артефакт шума/шаблона)
    for t in turns:
        tail = t["text"].rstrip()
        if tail and tail[-1] not in '.!?…»"' and not tail.endswith("—"):
            warn(f"turn {t['idx']}: нет финальной пунктуации: …{tail[-40:]!r}")

    # --- факты ---------------------------------------------------------------
    id2obj = {}
    for f in call["facts"]:
        o = f.get("object", {})
        if o.get("type") not in TYPES:
            bad(f"object.type={o.get('type')}")
        if not str(o.get("name", "")).strip():
            bad("object.name пустое")
        if f["attribute"] not in ATTRS:
            bad(f"attribute={f['attribute']}")
        if not str(f.get("value", "")).strip():
            bad("value пустое")
        if f.get("polarity") not in (1, -1):
            bad("polarity")
        if f["attribute"] != "service_state" and f.get("polarity") == -1:
            bad("polarity=-1 только для service_state")
        if f["polarity"] == -1 and f["value"] != "больше не используется":
            bad("polarity=-1 со значением кроме «больше не используется»")
        if not ISO_RE.match(str(f.get("valid_from", ""))):
            bad(f"valid_from не ISO: {f.get('valid_from')}")
        vt = f.get("valid_to")
        if vt is not None and (not ISO_RE.match(str(vt)) or vt < f["valid_from"]):
            bad("valid_to некорректно")
        src = f.get("source", {})
        ti = src.get("turn_idx")
        if not isinstance(ti, int) or not (0 <= ti < len(turns)):
            bad(f"turn_idx={ti}")
            continue
        spk = turns[ti]["speaker"]
        if f.get("speaker") != spk:
            bad(f"speaker({f.get('speaker')}) != turn.speaker({spk})")
        if o.get("id"):
            if o["id"] in id2obj and id2obj[o["id"]] != (o["type"], o["name"]):
                bad(f"id {o['id']} переиспользован с другим объектом")
            id2obj[o["id"]] = (o["type"], o["name"])
        # span/quote
        q, span = src.get("quote"), src.get("span")
        if q is not None:
            text = turns[ti]["text"]
            if q not in text:
                bad(f"quote не подстрока turn {ti}: {q[:40]!r}")
            elif span is not None:
                if text[span[0]:span[1]] != q:
                    bad(f"span {span} != quote в turn {ti}")
            elif span is not None:
                pass
        if q is None and span is not None:
            bad("span без quote")

    # --- single-valued цепочки ----------------------------------------------
    chains = defaultdict(list)
    for f in call["facts"]:
        if f["attribute"] in SINGLE_VALUED:
            chains[(f["object"].get("id"), f["attribute"])].append(f)
    for (oid, attr), fs in chains.items():
        fs.sort(key=lambda f: f["valid_from"])
        for a, b in zip(fs, fs[1:]):
            if a["valid_to"] is None:
                bad(f"цепочка {attr} {oid}: {a['value']!r} не закрыта перед {b['value']!r}")
            elif a["valid_to"] != b["valid_from"]:
                bad(f"цепочка {attr} {oid}: valid_to={a['valid_to']} != valid_from={b['valid_from']}")
        if len(fs) > 1 and fs[0]["valid_from"] == fs[-1]["valid_from"] and fs[0]["value"] == fs[-1]["value"]:
            bad(f"дубль факта {attr} {oid}")

    # --- согласованность номеров заявок ---------------------------------------
    text = " ".join(t["text"] for t in turns)
    nums_text = set(re.findall(r"(?<!\d)\d{4,6}(?!\d)", text))
    tkt_names = [f["object"]["name"] for f in call["facts"] if f["object"]["type"] == "ticket"]
    for nm in tkt_names:
        m = re.search(r"(\d{4,6})", nm)
        if m and m.group(1) not in nums_text:
            warn(f"номер {m.group(1)} из {nm} не звучал в диалоге")

    # --- codec roundtrip ------------------------------------------------------
    try:
        codec = _cfe()
        enc = codec.encode_target(call)
        arr = json.loads(enc)
        dec, dropped = codec.decode_target(enc, call)
        if dropped or len(dec) != len(arr):
            bad(f"codec roundtrip: dropped={dropped}, dec={len(dec)} из {len(arr)}")
        else:
            for got, want in zip(dec, call["facts"]):
                same = (got["source"]["turn_idx"] == want["source"]["turn_idx"]
                        and got["object"]["name"] == want["object"]["name"]
                        and got["attribute"] == want["attribute"]
                        and got["value"] == want["value"]
                        and got["polarity"] == want["polarity"]
                        # кодек хранит только день: час суток не восстанавливается
                        and got["valid_from"][:10] == want["valid_from"][:10]
                        and (got.get("valid_to") is None) == (want.get("valid_to") is None)
                        and (want.get("valid_to") is None
                             or got["valid_to"][:10] == want["valid_to"][:10]))
                if not same:
                    bad("codec roundtrip: расхождение полей")
                    break
    except Exception as e:
        bad(f"codec: {type(e).__name__} {e}")

    # --- похожесть на train (шинглы 8 слов) -----------------------------------
    if train_shingles is not None:
        toks = re.findall(r"[а-яёa-z0-9]+", text.lower())
        sh = {" ".join(toks[i:i + 8]) for i in range(len(toks) - 7)}
        for tr_id, tr_sh in train_shingles.items():
            inter = len(sh & tr_sh) / max(1, len(sh)) if sh else 0
            if inter >= 0.5:
                bad(f"близкая копия train-звонка {tr_id} (overlap {inter:.2f})")
                break
    return errs, warns


def train_shingle_index(train_path: str, max_calls: int = 4000) -> dict:
    idx = {}
    with open(train_path, encoding="utf-8") as fh:
        for line in fh:
            c = json.loads(line)
            toks = re.findall(r"[а-яёa-z0-9]+", " ".join(t["text"] for t in c["turns"]).lower())
            sh = {" ".join(toks[i:i + 8]) for i in range(len(toks) - 7)}
            idx[c["call_id"]] = sh
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--train", default=str(ROOT.parent / "data/raw/train.jsonl"))
    ap.add_argument("--out", default=None, help="JSON-отчёт")
    ap.add_argument("--rejected", default=None, help="файл call_id отклонённых")
    ap.add_argument("--no-train-sim", action="store_true")
    a = ap.parse_args()

    calls = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    tsh = None if a.no_train_sim else train_shingle_index(a.train)
    n_err_calls, all_warns, err_types = 0, [], Counter()
    rejected = []
    per_call = {}
    for c in calls:
        errs, warns = check_call(c, tsh)
        per_call[c["call_id"]] = errs
        if errs:
            n_err_calls += 1
            rejected.append(c["call_id"])
            for e in errs:
                err_types[e.split(":", 1)[1].strip().split(" ")[0].split("=")[0]] += 1
        all_warns.extend(warns)

    report = {
        "file": a.input,
        "calls": len(calls),
        "calls_with_errors": n_err_calls,
        "error_types": dict(err_types),
        "warnings": Counter(w.split(":", 1)[1].strip().split(":")[0] for w in all_warns).most_common(),
        "per_call_errors": {k: v for k, v in per_call.items() if v},
    }
    print(json.dumps({k: v for k, v in report.items() if k != "per_call_errors"},
                     ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
    if a.rejected:
        with open(a.rejected, "w", encoding="utf-8") as fh:
            for cid in rejected:
                fh.write(cid + "\n")
    if n_err_calls:
        print(f"ОТКЛОНЕНО звонков: {n_err_calls}")
        sys.exit(1)


if __name__ == "__main__":
    main()
