"""Простановка дат фактам (проверено стендом date_harness.py на train-срезе, не на val).

  * договорённость / состояние / тариф: первая относительная дата в реплике-якоре (для договорённости —
    будущий день недели, для остального — прошедший); без даты — момент звонка;
  * состояние и тариф выстраиваются в цепочку по дате: valid_to предыдущего = valid_from следующего;
  * статусы заявки — «история»: статусы упорядочены (принята → в работе → выезд → выполнена),
    к каждому привязывается ближайшая дата из его клаузы; «до X» задаёт конец статуса, а не начало;
    конец предыдущего статуса = начало следующего и наоборот; у последнего статуса без даты — день звонка.
"""
import re
from datetime import datetime
import dates as DT

MENTIONS_FROM = False  # искать начало статуса по всем уверенным упоминаниям
MENTIONS_TO = True     # искать конец статуса по всем уверенным упоминаниям
STATUS_RANK = {"принята": 0, "в работе": 1, "назначен выезд специалиста": 2, "выполнена": 3}
CLAUSE_SPLIT = re.compile(r"[.!?;…]|\s—\s|,\s+а\s+|\sа\s+(?:сейчас|теперь|сегодня|потом|затем|вчера)|,\s+(?:но|потом|затем|а)\s")
TO_CUE = re.compile(r"(?:\bдо(?:\s+самого)?\s*$|\bвплоть\s+до\s*$|\bпо\s*$)")


def status_rank(v):
    v = v.lower()
    if "выполн" in v or "закрыт" in v: return 3
    if "выезд" in v: return 2
    if "работ" in v: return 1
    if "принят" in v: return 0
    return 1


def clauses(text):
    out, s = [], 0
    for m in CLAUSE_SPLIT.finditer(text):
        out.append((s, m.start())); s = m.end()
    out.append((s, len(text)))
    return out


def attached(text, pos, call_dt, future=False):
    """(role, date) ближайшей к pos даты в той же клаузе (или в пределах 40 символов); role ∈ {from, to}"""
    ex = DT.resolve(text, call_dt)
    if not ex:
        return None
    cl = next(((a, b) for a, b in clauses(text) if a <= pos <= b), (0, len(text)))
    cand = [e for e in ex if cl[0] <= e[2] <= cl[1]] or [e for e in ex if abs(e[2] - pos) <= 40]
    if not cand:
        return None
    kind, x, p = min(cand, key=lambda e: abs(e[2] - pos))
    if isinstance(x, tuple):
        x = x[2] if future else x[1]
    role = "to" if TO_CUE.search(text[max(0, p - 20):p].lower()) else "from"
    return role, x


def first_date(text, call_dt, future):
    dd = DT.pick(DT.resolve(text, call_dt), call_dt.date(), future)
    return DT.iso(dd) if dd else None


def set_dates(d, facts, key_fn=lambda f: f["object"].get("name", ""), agreement_from=None):
    """agreement_from(f) → iso-день или None: внешний (обученный) выбор даты для договорённостей"""
    call_dt = datetime.fromisoformat(d["started_at"]); today = call_dt.date().isoformat() + "T00:00:00"
    turn = lambda f: d["turns"][f["source"]["turn_idx"]]["text"]
    for f in facts:
        f["valid_to"] = None
        f["_explicit"] = False
        if f["attribute"] == "agreement" and agreement_from:
            day = agreement_from(f)
            f["valid_from"] = d["started_at"] if not day or day == call_dt.date().isoformat() else day + "T00:00:00"
        elif f["attribute"] != "ticket_status":
            fd = first_date(turn(f), call_dt, f["attribute"] == "agreement")
            f["valid_from"] = fd or d["started_at"]
            f["_explicit"] = fd is not None
    # состояние / тариф: недатированное значение — более раннее (старый тариф, «работало нормально»),
    # конец предыдущего = явная дата начала следующего
    chains = {}
    for f in facts:
        if f["attribute"] in ("service_state", "tariff"):
            chains.setdefault((f["attribute"], key_fn(f)), []).append(f)
    for (a, _), fs in chains.items():
        fs.sort(key=lambda f: (f["_explicit"], f["valid_from"][:10], f["source"]["turn_idx"]))
        for p, n in zip(fs, fs[1:]):
            if n["_explicit"] and n["valid_from"][:10] > (p["valid_from"][:10] if p["_explicit"] else "0000"):
                p["valid_to"] = n["valid_from"][:10] + "T00:00:00"
    for f in facts:
        f.pop("_explicit", None)
    # статусы заявки: история
    tickets = {}
    for f in facts:
        if f["attribute"] == "ticket_status":
            tickets.setdefault(key_fn(f), []).append(f)
    for _, fs in tickets.items():
        fs.sort(key=lambda f: (status_rank(f["value"]), f["source"]["turn_idx"]))
        fr, to = [], []
        for f in fs:
            # дата может быть при любом уверенном упоминании статуса, не только при первом
            ments = f.get("_mentions") or [(f["source"]["turn_idx"], f.get("_pos") or 0)]
            atts = [attached(d["turns"][ti]["text"], pos, call_dt) for ti, pos in ments]
            a0 = atts[[m[0] for m in ments].index(f["source"]["turn_idx"])] if f["source"]["turn_idx"] in [m[0] for m in ments] else None
            pool_f = atts if MENTIONS_FROM else [a0]
            pool_t = atts if MENTIONS_TO else [a0]
            ef = next((a[1] for a in pool_f if a and a[0] == "from"), None)
            et = next((a[1] for a in pool_t if a and a[0] == "to"), None)
            fr.append(DT.iso(ef) if ef else None)
            to.append(DT.iso(et) if et else None)
        for i in range(1, len(fs)):            # начало следующего = явный конец предыдущего
            if fr[i] is None and to[i - 1]:
                fr[i] = to[i - 1]
        if fr[-1] is None:                     # текущий статус без даты — с сегодняшнего дня
            fr[-1] = today
        for i in range(len(fs) - 1):           # конец предыдущего = начало следующего (или сегодня)
            if to[i] is None:
                to[i] = fr[i + 1] or today
        for f, a, b in zip(fs, fr, to):
            f["valid_from"] = a or d["started_at"]
            f["valid_to"] = b if b and (a is None or b[:10] > a[:10]) else None
    return facts
