"""Детерминированные правила CFE: смещение даты начала факта (текст + статистика train) и полярность. Только stdlib."""
import json
import os
import re
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec  # noqa: E402
from textsim import _sim, _tokens  # noqa: E402

NUM = {"один": 1, "одного": 1, "одну": 1, "два": 2, "две": 2, "двух": 2, "три": 3, "трех": 3,
       "четыре": 4, "четырех": 4, "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7,
       "восемь": 8, "девять": 9, "десять": 10, "пару": 2}
WEEKDAY = {"понедельник": 0, "понедельника": 0, "вторник": 1, "вторника": 1, "среда": 2, "среду": 2,
           "среды": 2, "четверг": 3, "четверга": 3, "пятница": 4, "пятницу": 4, "пятницы": 4,
           "суббота": 5, "субботу": 5, "субботы": 5, "воскресенье": 6, "воскресенья": 6}
SINGLE = {"вчера": -1, "позавчера": -2, "завтра": 1, "послезавтра": 2, "сегодня": 0}
MORPH = False          # A/B: без выигрыша, pymorphy3 не нужен
SAME_SENTENCE = True   # A/B: +0.0006 val, +0.0005 val_noisy
DATE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "date_model.json")
WORD_RE = re.compile(r"[а-яёa-z0-9]+")


def _norm(text):
    return text.lower().replace("ё", "е")


def _fz(w, lex, th):
    best, bs = None, 0.0
    forms = (w, _lemma(w)) if MORPH else (w,)
    for k in lex:
        s = max(_sim(f, k) for f in forms)
        if s > bs:
            best, bs = k, s
    return best if bs >= th else None


_MORPH = None


def _lemma(w):
    global _MORPH
    if _MORPH is None:
        import pymorphy3
        _MORPH = pymorphy3.MorphAnalyzer()
    return _MORPH.parse(w)[0].normal_form.replace("ё", "е")


def _num(w):
    if w.isdigit():
        return int(w)
    k = _fz(w, NUM, 0.8)
    return NUM[k] if k else None


def _month_back(started_at):
    return int(re.search(r"месяц назад (-\d+)", codec.date_header(started_at)).group(1))


def date_mentions(text, started_at, future):
    """[(смещение в днях, позиция символа, вид)] для относительных выражений даты в тексте.
    Вид: day1 (вчера/завтра…), ago (… назад), thru (через …), last (прошлой …), wd (день недели)."""
    toks = [(m.group(0), m.start()) for m in WORD_RE.finditer(_norm(text))]
    W = [w for w, _ in toks]
    wd = date.fromisoformat(started_at[:10]).weekday()
    out = []
    for i, (w, p) in enumerate(toks):
        pv = W[i - 1] if i else ""
        nx = W[i + 1] if i + 1 < len(W) else ""
        nx2 = W[i + 2] if i + 2 < len(W) else ""
        k = _fz(w, SINGLE, 0.85)
        if k:
            out.append((SINGLE[k], p, "day1"))
            continue
        if _fz(w, ["назад"], 0.8):
            if pv and _fz(pv, ["неделю"], 0.75):
                out.append((-7, p, "ago"))
                continue
            if pv and _fz(pv, ["месяц"], 0.75):
                out.append((_month_back(started_at), p, "ago"))
                continue
            if pv and (re.match(r"(дн|ден|сут)", pv) or _fz(pv, ["дня", "дней", "день"], 0.75)):
                n = _num(W[i - 2]) if i >= 2 else None
                out.append((-(n or 1), p, "ago"))
                continue
        if _fz(w, ["через"], 0.8):
            n = _num(nx) if nx else None
            if n and (re.match(r"(дн|ден)", nx2) or _fz(nx2, ["дня", "дней"], 0.75)):
                out.append((n, p, "thru"))
                continue
            if nx and _fz(nx, ["неделю"], 0.75):
                out.append((7, p, "thru"))
                continue
            if nx and _fz(nx, ["день"], 0.75):
                out.append((1, p, "thru"))
                continue
        if _fz(w, ["прошлой", "прошлом"], 0.8):
            if nx and _fz(nx, ["неделе"], 0.75):
                out.append((-7, p, "last"))
                continue
            if nx and _fz(nx, ["месяце"], 0.75):
                out.append((_month_back(started_at), p, "last"))
                continue
        k = _fz(w, WEEKDAY, 0.8) if len(w) >= 5 else None
        if k:
            d = WEEKDAY[k]
            out.append(((((d - wd) % 7) or 7) if future else -(((wd - d) % 7) or 7), p, "wd"))
    return out


def _anchor(text, value):
    """Позиция слова реплики, ближайшего к значению факта (сравнение первых 6 букв), или None."""
    vt = [v for v in _tokens(value) if len(v) >= 4]
    best, bs = None, 0.0
    for m in WORD_RE.finditer(_norm(text)):
        for v in vt:
            s = _sim(m.group(0)[:6], v[:6])
            if s > bs:
                best, bs = m.start(), s
    return best if bs >= 0.8 else None


def pick_mention(call, t, value, attr):
    """(смещение, вид) выражения даты, относящегося к value; (0, "none"), если выражений нет."""
    text = call["turns"][t]["text"]
    ex = date_mentions(text, call["started_at"], attr == "agreement")
    if not ex:
        return 0, "none"
    a = _anchor(text, value)
    if a is None:
        return ex[0][0], ex[0][2]
    if SAME_SENTENCE:
        cuts = [0] + [m.end() for m in re.finditer(r"[.!?;]", text)] + [len(text) + 1]
        lo, hi = max(c for c in cuts if c <= a), min(c for c in cuts if c > a)
        ex = [e for e in ex if lo <= e[1] < hi] or ex
    o, _, k = min(ex, key=lambda e: abs(e[1] - a))
    return o, k


NEG_VALUE = "больше не используется"


def polarity(attr, value):
    """В корпусе −1 ровно у service_state = «больше не используется»."""
    return -1 if attr == "service_state" and value == NEG_VALUE else 1


SOURCES = ("rules", "prior", "zero")  # порядок = приоритет при ничьей


def _prior_key(attr, value, kind):
    return f"{attr}|{value}|{int(kind != 'none')}"


def _prior(prior, attr, value, kind):
    return prior.get(_prior_key(attr, value, kind), prior.get(_prior_key(attr, "*", kind), 0))


def build_date_model(calls):
    """prior: самое частое смещение по (атрибут, value, есть выражение) и запасное (атрибут, *, есть выражение);
    policy: для ячейки «атрибут|вид» источник с наибольшим числом верных дней."""
    from collections import Counter, defaultdict
    rows, tab = [], defaultdict(Counter)
    for c in calls:
        for f in c["facts"]:
            a, t, v = f["attribute"], f["source"]["turn_idx"], f["value"]
            g = codec.day_offset(c["started_at"], f["valid_from"])
            o, k = pick_mention(c, t, v, a)
            rows.append((a, v, k, o, g))
            tab[_prior_key(a, v, k)][g] += 1
            tab[_prior_key(a, "*", k)][g] += 1
    prior = {key: cnt.most_common(1)[0][0] for key, cnt in tab.items()}
    score = defaultdict(Counter)
    for a, v, k, o, g in rows:
        cand = {"rules": o, "prior": _prior(prior, a, v, k), "zero": 0}
        for s in SOURCES:
            score[f"{a}|{k}"][s] += cand[s] == g
    policy = {cell: max(SOURCES, key=lambda s: (sc[s], -SOURCES.index(s))) for cell, sc in score.items()}
    return {"prior": prior, "policy": policy}


TUNE_CALLS = 1200


def tune_policy(model, calls):
    """Политика по реальной метрике: valid_from и valid_to вместе (temporal_recall на oracle), а не только день valid_from.
    Старт — «rules» везде; покоординатно для каждой ячейки пробуются остальные источники, берётся улучшение."""
    import random
    import case_metrics, clf_eval, clf_task as T
    sub = random.Random(0).sample(calls, min(TUNE_CALLS, len(calls)))
    cat = T.load_catalog()
    vl = T.value_labels(cat)
    gold = {c["call_id"]: c for c in sub}

    def score(m):
        return case_metrics.evaluate(gold, clf_eval.oracle(sub, cat, vl, m))["temporal_recall"]

    model = {**model, "policy": {cell: "rules" for cell in model["policy"]}}
    best = score(model)
    for cell in sorted(model["policy"]):
        for s in SOURCES[1:]:
            cand = {**model, "policy": {**model["policy"], cell: s}}
            sc = score(cand)
            if sc > best:
                best, model = sc, cand
    print("tuned temporal_recall on train subsample:", best)
    return model


def load_date_model(path=DATE_MODEL_PATH):
    return json.load(open(path, encoding="utf-8"))


def resolve_date(dm, call, t, value, attr):
    """Смещение valid_from в днях от дня звонка по политике ячейки; неизвестная ячейка -> правило текста."""
    o, k = pick_mention(call, t, value, attr)
    src = dm["policy"].get(f"{attr}|{k}", "rules")
    return {"rules": o, "prior": _prior(dm["prior"], attr, value, k), "zero": 0}[src]


# Цепочка статусов заявки: в эталоне train (2677 фактов в цепочках из ≥2 статусов) порядок статусов фиксирован,
# valid_from соседних статусов отличается на 1–2 дня. Явная дата (день недели, «N назад», «вчера» у не первого
# статуса) надёжна, остальные смещения выводятся из соседа. Таблицы подобраны на train по паре (valid_from, valid_to).
STATUS_ORDER = {"принята": 0, "в работе": 1, "назначен выезд специалиста": 2, "выполнена": 3}
CHAIN_BACK = {"none": 1, "last": 2, "day1": 3, "wd": 2, "ago": 2, "thru": 2}  # от следующего статуса назад
CHAIN_FWD = {"none": 1, "last": 2, "day1": 2, "wd": 2, "ago": 2, "thru": 2}   # от предыдущего статуса вперёд


def chain_offsets(items):
    """items: [(смещение, вид)] статусов одной заявки в хронологическом порядке -> новые смещения."""
    n = len(items)
    v = [o for o, _ in items]
    if n < 2:
        return v
    ks = [k for _, k in items]
    known = [k in ("wd", "ago") or (k == "day1" and (i > 0 or i == n - 1)) for i, k in enumerate(ks)]
    for i in range(1, n):
        if not known[i] and known[i - 1]:
            v[i], known[i] = v[i - 1] + CHAIN_FWD[ks[i]], True
    for i in range(n - 2, -1, -1):
        if not known[i]:
            v[i], known[i] = v[i + 1] - CHAIN_BACK[ks[i]], True
    return v


if __name__ == "__main__":
    train = [json.loads(l) for l in open(os.path.join(os.path.dirname(DATE_MODEL_PATH), "raw", "train.jsonl"), encoding="utf-8")]
    model = tune_policy(build_date_model(train), train)
    json.dump(model, open(DATE_MODEL_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(model["policy"], ensure_ascii=False, indent=1))
