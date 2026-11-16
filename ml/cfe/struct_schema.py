"""Структурная схема GLiNER2.5 для CFE и сборка фактов из сырых ответов модели. Только stdlib + cfe."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textsim  # noqa: E402
import clf_task  # noqa: E402
import codec  # noqa: E402
import objects  # noqa: E402
import rules  # noqa: E402

TYPES = ["service", "device", "ticket", "client", "other"]
ATTRS = ["service_state", "tariff", "ticket_status", "action", "agreement"]
POL = ["approval", "denial"]
GROUPS = {"attribute": ATTRS, "type": TYPES, "polarity": POL}
FACT_FIELDS = ["value", "attribute", "type", "polarity", "object", "date"]


def qlabel(group, label):
    return f"{group}: {label}"


WIN_RE = re.compile(r"[а-яёa-z0-9]+", re.I)
TRAIN_MIN_SIM = 0.6


def build_schema(extractor, kind="entities"):
    if kind == "fact":
        return (extractor.create_schema()
                .structure("fact", mode="natural", anchor="value")
                .field("value", dtype="str")
                .field("attribute", dtype="str", choices=ATTRS)
                .field("type", dtype="str", choices=TYPES)
                .field("polarity", dtype="str", choices=POL)
                .field("object", dtype="str")
                .field("date", dtype="str")
                .entities(["person"]))
    if kind != "entities":
        raise ValueError(f"unknown schema kind {kind!r}")
    from gliner2 import AttributeGroup
    return (extractor.create_schema()
            .entities(["value", "person", "date"])
            .entity_attributes({g: AttributeGroup(labels, applies_to=["value"], qualify_labels=True)
                                for g, labels in GROUPS.items()}))


def turn_offset(ctx, call, t):
    sp = "К" if call["turns"][t]["speaker"] == "client" else "О"
    return ctx.index(f">>> [{t}] ") + len(f">>> [{t}] {sp}: ")


def locate_value(text, value):
    """Лучшее окно из n-1..n+1 слов text по _value_sim к value -> (score, start, end)."""
    ws = [(m.start(), m.end()) for m in WIN_RE.finditer(text)]
    if not ws:
        return 0.0, None, None
    n = max(1, len(textsim._tokens(value)))
    best = (0.0, None, None)
    for k in range(max(1, n - 1), n + 2):
        for i in range(0, len(ws) - k + 1):
            s, e = ws[i][0], ws[i + k - 1][1]
            sc = textsim._value_sim(value, text[s:e])
            if sc > best[0]:
                best = (sc, s, e)
    if best[1] is None:
        s, e = ws[0][0], ws[0][1]
        return 0.0, s, e
    return best


def _f(v):
    """Нормализатор поля вывода -> (text, confidence, start, end) или None."""
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None or v == "":
        return None
    if isinstance(v, dict):
        txt = v.get("text") if v.get("text") is not None else v.get("label")
        if txt in (None, ""):
            return None
        return (txt, v.get("confidence", 1.0), v.get("start"), v.get("end"))
    return (v, 1.0, None, None)


def _value_items(raw):
    """Кандидаты value в едином виде (item, date_span, object_span)."""
    if raw is None:
        return
    if "fact" in raw:
        for rec in raw["fact"] or []:
            if not isinstance(rec.get("value"), dict):
                continue
            item = dict(rec["value"])
            for g in ("attribute", "type", "polarity"):
                c = _f(rec.get(g))
                item[g] = {"label": c[0]} if c else {}
            dsp, osp = _f(rec.get("date")), _f(rec.get("object"))
            yield item, (dsp if dsp and dsp[2] is not None else None), (osp if osp and osp[2] is not None else None)
    else:
        for item in raw.get("entities", {}).get("value", []) or []:
            yield item, None, None


SNAP_STOP = {"уже", "раза", "раз", "три", "два", "еще", "все", "вот", "ну", "я", "у", "меня", "мне",
             "с", "со", "в", "на", "и", "а", "по", "так", "это", "же", "бы"}


def _stem_eq(a, b):
    """Одно слово: общий префикс ≥ min(5, длины) (перезагружал ~ перезагрузка) или посимвольно ≥ 0.75 (ASR-опечатки)."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= min(5, len(a), len(b)) or textsim._sim(a, b) >= 0.75


def _bag_sim(a, b):
    """Dice по словам без учёта порядка: 2·совпавших / (|a| + |b|), служебные слова отброшены."""
    ta = [w for w in textsim._tokens(a) if w not in SNAP_STOP]
    tb = [w for w in textsim._tokens(b) if w not in SNAP_STOP]
    if not ta or not tb:
        return 0.0
    used, hits = set(), 0
    for w in ta:
        for j, v in enumerate(tb):
            if j not in used and _stem_eq(w, v):
                used.add(j)
                hits += 1
                break
    return 2 * hits / (len(ta) + len(tb))


def snap(attr, text, vl, th):
    """Ближайшая метка vl[attr] по max(_value_sim, _bag_sim); ничья — по _value_sim, затем первая метка. Ниже th — сырой текст."""
    best, key = None, None
    for lab in vl.get(attr, []):
        o = textsim._value_sim(text, lab)
        k = (max(o, _bag_sim(text, lab)), o)
        if key is None or k > key:
            best, key = lab, k
    return best if best is not None and key[0] >= th else text


def date_spans(text, started_at, future):
    """[(смещение, start, end, kind)] минимальные окна слов вокруг ключевого слова выражения даты."""
    ws = [(m.start(), m.end()) for m in WIN_RE.finditer(text)]
    out = []
    for o, p, k in rules.date_mentions(text, started_at, future):
        idx = [i for i, w in enumerate(ws) if w[0] == p]
        if not idx:
            continue
        i = idx[0]
        wins = sorted(((b + a, b, a) for b in range(3) for a in range(3)))
        for _, b, a in wins:
            lo, hi = i - b, i + a
            if lo < 0 or hi >= len(ws):
                continue
            s, e = ws[lo][0], ws[hi][1]
            if any(m[0] == o for m in rules.date_mentions(text[s:e], started_at, future)):
                out.append((o, s, e, k))
                break
    return out


def model_offset(dm, call, t, value, attr, dspans, s):
    """Смещение в днях по спанам date, найденным моделью, и политике date_model."""
    text = call["turns"][t]["text"]
    ex = []
    for ds, de in dspans:
        ms = rules.date_mentions(text[ds:de], call["started_at"], attr == "agreement")
        if ms:
            ex.append((ms[0][0], ds, ms[0][2]))
    if not ex:
        o, k = 0, "none"
    else:
        cuts = [0] + [m.end() for m in re.finditer(r"[.!?;]", text)] + [len(text) + 1]
        lo, hi = max(c for c in cuts if c <= s), min(c for c in cuts if c > s)
        ex = [e for e in ex if lo <= e[1] < hi] or ex
        o, _, k = min(ex, key=lambda e: abs(e[1] - s))
    src = dm["policy"].get(f"{attr}|{k}", "rules")
    return {"rules": o, "prior": rules._prior(dm["prior"], attr, value, k), "zero": 0}[src]


def call_facts(call, raws, offs, vl, dm, tau, snap_th, date_tau=None, llm=None, keep_conf=False):
    started = call["started_at"]
    name, nconf = "", -1.0
    for r in raws:
        for p in (r or {}).get("entities", {}).get("person", []) or []:
            f = _f(p)
            if f and f[1] > nconf:
                name, nconf = f[0], f[1]
    cands = {}
    for t, raw in enumerate(raws):
        dabs = []
        if date_tau is not None:
            for d in (raw or {}).get("entities", {}).get("date", []) or []:
                f = _f(d)
                if f and f[1] >= date_tau and f[2] is not None and f[3] is not None:
                    dabs.append((f[2], f[3]))
        text = call["turns"][t]["text"]
        for item, dsp, osp in _value_items(raw):
            v = _f(item)
            if not v or not isinstance(item, dict):
                continue
            _, conf, st, en = v
            if st is None or en is None or not (offs[t] <= st and en <= offs[t] + len(text)):
                continue
            attr = (item.get("attribute") or {}).get("label")
            if attr not in ATTRS:
                continue
            th = tau.get(attr, 1.0) if isinstance(tau, dict) else tau
            if conf < th:
                continue
            o = None
            if llm is not None:
                o = llm.get((t, st, en, attr))
                if o is None or o == "НЕТ":
                    continue
            ty = (item.get("type") or {}).get("label")
            ty = ty if ty in TYPES else "client"
            denial = (item.get("polarity") or {}).get("label") == "denial"
            s, e = st - offs[t], en - offs[t]
            if attr == "service_state" and denial:
                value, polr = rules.NEG_VALUE, -1
            else:
                value, polr = snap(attr, o if o is not None else text[s:e], vl, snap_th), 1
            nm = ""
            if ty == "client":
                nm = name
            elif ty == "ticket":
                nm = clf_task.ticket_name(call, t)
            elif ty in ("service", "device"):
                if osp is not None and osp[1] >= 0.5:
                    nm = objects.snap_name(osp[0], ty)
                if not nm:
                    nm = objects.object_name(call, t, ty, s)
            if date_tau is None:
                k = rules.resolve_date(dm, call, t, value, attr)
            else:
                if dsp is not None:
                    ok = dsp[1] >= date_tau and offs[t] <= dsp[2] and dsp[3] <= offs[t] + len(text)
                    dspans = [(dsp[2] - offs[t], dsp[3] - offs[t])] if ok else []
                else:
                    dspans = [(a - offs[t], b - offs[t]) for a, b in dabs
                              if offs[t] <= a and b <= offs[t] + len(text)]
                k = model_offset(dm, call, t, value, attr, dspans, s)
            fact = {
                "object": {"type": ty, "name": nm},
                "attribute": attr, "value": value, "polarity": polr,
                "speaker": call["turns"][t]["speaker"],
                "valid_from": codec.iso(started, k),
                "valid_to": None,
                "source": {"turn_idx": t, "span": [s, e], "quote": text[s:e]},
            }
            key = (t, attr, value)
            if key not in cands or conf > cands[key][0]:
                cands[key] = (conf, fact)
    if keep_conf:
        for conf, f in cands.values():
            f["conf"] = conf
    facts = codec.drop_repeated_states([f for _, f in cands.values()])
    if date_tau is None:
        _chain_ticket_dates(dm, call, facts)
    return codec.close_states(facts)


def _chain_ticket_dates(dm, call, facts):
    """Даты статусов заявки по цепочке (rules.chain_offsets); порядок — по каноническому порядку статусов, затем по реплике."""
    by = {}
    for f in facts:
        if f["attribute"] == "ticket_status":
            by.setdefault(f["object"]["name"].lower(), []).append(f)
    for chain in by.values():
        if len(chain) < 2:
            continue
        chain.sort(key=lambda f: (rules.STATUS_ORDER.get(f["value"], 9), f["source"]["turn_idx"], f["source"]["span"][0]))
        items = [(rules.resolve_date(dm, call, f["source"]["turn_idx"], f["value"], "ticket_status"),
                  rules.pick_mention(call, f["source"]["turn_idx"], f["value"], "ticket_status")[1]) for f in chain]
        for f, o in zip(chain, rules.chain_offsets(items)):
            f["valid_from"] = codec.iso(call["started_at"], o)
