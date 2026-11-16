"""Контролируемая аугментация реальных звонков train (подход B).

Преобразования гарантированно сохраняют корректность разметки:
  * date_shift      — сдвиг звонка на целое число недель (дни недели в тексте
                      остаются валидными), все даты фактов сдвигаются вместе;
  * renumber        — замена номеров заявок на другие той же длины
                      (символьные оффсеты не меняются);
  * tariff_rename   — замена названия тарифа с перепривязкой span/quote;
  * filler_insert   — вставка несущественной реплики с пересчётом turn_idx;
  * paraphrase      — лёгкие синонимические замены ТОЛЬКО в репликах, на которые
                      не ссылается ни один факт;
  * asr_noise       — коррупция текста (cfe/asr_noise) со сбросом span/quote
                      в null (в gold 45% фактов и так без source-текста).

Любое преобразование, которое не удаётся применить безопасно, отклоняет пример.
"""
from __future__ import annotations

import copy
import random
import re
from datetime import date, timedelta

TARIFF_RE = None  # ленивая загрузка каталога


def _catalog():
    import json
    from .catalog import CATALOG_PATH
    with open(CATALOG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
def _reground(call: dict) -> bool:
    """После правок текста пересчитать span/quote. False — если цитата потеряна."""
    for f in call["facts"]:
        src = f["source"]
        q = src.get("quote")
        if not q:
            continue
        t = call["turns"][src["turn_idx"]]["text"]
        hits = [m.start() for m in re.finditer(re.escape(q), t)]
        if len(hits) == 1 and src.get("span") is not None:
            src["span"] = [hits[0], hits[0] + len(q)]
        elif len(hits) == 1:
            src["span"] = [hits[0], hits[0] + len(q)]
        elif len(hits) == 0:
            # цитата могла быть изменена правкой — пытаемся нечёткий поиск
            near = _fuzzy_find(q, t)
            if near is not None:
                src["quote"], src["span"] = near[0], [near[1], near[1] + len(near[0])]
            else:
                src["quote"], src["span"] = None, None
        else:
            src["span"] = None
    return True


def _fuzzy_find(quote: str, text: str):
    """Найти наиболее похожий фрагмент text для quote (SequenceMatcher)."""
    from difflib import SequenceMatcher
    best, bs = None, 0.0
    words = quote.split(" ")
    lo = max(1, len(words) - 3)
    hi = len(words) + 3
    toks = text.split(" ")
    for i in range(len(toks)):
        for n in range(lo, min(hi, len(toks) - i) + 1):
            frag = " ".join(toks[i:i + n])
            s = SequenceMatcher(None, quote, frag).ratio()
            if s > bs:
                best, bs = frag, s
    if best is not None and bs >= 0.85:
        pos = text.find(best)
        return best, pos
    return None


# --------------------------------------------------------------------------- #
def date_shift(call: dict, weeks: int) -> dict | None:
    """Сдвиг всей истории на weeks недель. Текст не содержит абсолютных дат."""
    c = copy.deepcopy(call)
    txt = " ".join(t["text"] for t in c["turns"])
    if re.search(r"\d{1,2}[./]\d{1,2}(\.\d{2,4})?|\b(19|20)\d{2}\b", txt):
        return None  # в тексте есть абсолютные даты — сдвиг небезопасен
    delta = timedelta(days=7 * weeks)
    c["started_at"] = (date.fromisoformat(c["started_at"][:10]) + delta).isoformat() + c["started_at"][10:]
    for f in c["facts"]:
        for key in ("valid_from", "valid_to"):
            v = f.get(key)
            if v:
                f[key] = (date.fromisoformat(v[:10]) + delta).isoformat() + v[10:]
    return c


def renumber_tickets(call: dict, rng: random.Random) -> dict | None:
    """Номера заявок -> новые той же длины (оффсеты не смещаются)."""
    c = copy.deepcopy(call)
    nums = set()
    for f in c["facts"]:
        if f["object"]["type"] == "ticket":
            m = re.search(r"(\d{3,6})", f["object"]["name"])
            if m:
                nums.add(m.group(1))
        m = re.search(r"(\d{3,6})", f["value"])
        if m and "заявк" in f["value"].lower():
            nums.add(m.group(1))
    nums = {n for n in nums if 3 <= len(n) <= 6}
    if not nums:
        return None
    mapping = {}
    for n in nums:
        for _ in range(50):
            new = "".join(rng.choice("0123456789") if i else rng.choice("123456789")
                          for i in range(len(n)))
            if new != n and new not in mapping.values():
                mapping[n] = new
                break
        if n not in mapping:
            return None

    def sub(m):
        return mapping.get(m.group(0), m.group(0))

    pat = re.compile(r"(?<!\d)(" + "|".join(sorted(mapping, key=len, reverse=True)) + r")(?!\d)")
    for t in c["turns"]:
        t["text"] = pat.sub(sub, t["text"])
    for f in c["facts"]:
        if f["object"]["type"] == "ticket":
            f["object"]["name"] = pat.sub(sub, f["object"]["name"])
        f["value"] = pat.sub(sub, f["value"])
        q = f["source"].get("quote")
        if q:
            f["source"]["quote"] = pat.sub(sub, q)
    _reground(c)
    return c


def tariff_rename(call: dict, rng: random.Random) -> dict | None:
    """Замена одного тарифа на другой (перепривязка цитат)."""
    c = copy.deepcopy(call)
    tariffs = _catalog()["tariffs"]
    old_names = set()
    for f in c["facts"]:
        if f["attribute"] == "tariff" and f["value"] in tariffs:
            old_names.add(f["value"])
    old_names = sorted(old_names, key=len, reverse=True)
    if not old_names:
        return None
    old = old_names[0]
    for _ in range(50):
        new = rng.choice(tariffs)
        if new != old and new not in old_names:
            break
    else:
        return None
    pat = re.compile(re.escape(old))
    for t in c["turns"]:
        t["text"] = pat.sub(new, t["text"])
    for f in c["facts"]:
        f["value"] = pat.sub(new, f["value"])
        if f["object"]["type"] == "client" and old in f["object"].get("name", ""):
            f["object"]["name"] = pat.sub(new, f["object"]["name"])
        q = f["source"].get("quote")
        if q and old in q:
            f["source"]["quote"] = pat.sub(new, q)
    _reground(c)
    return c


FILL_O = ["Так, одну секундочку, открываю карточку…", "Понимаю вас, продолжайте.",
          "Так-так, вижу ваш аккаунт.", "Да, я записываю."]
FILL_C = ["Да уж, сил нет.", "Вот именно.", "Извините, что повторяюсь."]


def filler_insert(call: dict, rng: random.Random) -> dict | None:
    """Вставка несущественной реплики перед последней или предпоследней."""
    c = copy.deepcopy(call)
    turns = c["turns"]
    if len(turns) < 4:
        return None
    pos = None
    for k in rng.sample(range(1, len(turns)), min(4, len(turns) - 1)):
        prev_spk = turns[k - 1]["speaker"]
        nxt_spk = turns[k]["speaker"] if k < len(turns) else None
        cand = "operator" if prev_spk == "client" else "client"
        if cand != nxt_spk:
            pos, spk = k, cand
            break
    if pos is None:
        return None
    pool = FILL_O if spk == "operator" else FILL_C
    turns.insert(pos, {"idx": 0, "speaker": spk, "text": rng.choice(pool)})
    for i, t in enumerate(turns):
        t["idx"] = i
    for f in c["facts"]:
        if f["source"]["turn_idx"] >= pos:
            f["source"]["turn_idx"] += 1
    return c


SYNONYM = [
    (re.compile(r"\bЗдравствуйте\b"), "Добрый день"),
    (re.compile(r"\bДобрый день\b"), "Здравствуйте"),
    (re.compile(r"\bСпасибо\b"), "Благодарю"),
    (re.compile(r"\bБлагодарю\b"), "Спасибо"),
    (re.compile(r"\bХорошо\b"), "Ок"),
    (re.compile(r"\bПонял\b"), "Ясно"),
    (re.compile(r"\bПонимаю\b"), "Ясно"),
    (re.compile(r"\bпроверить\b"), "посмотреть"),
    (re.compile(r"\bсейчас\b"), "прямо сейчас"),
    (re.compile(r"\bконечно\b"), "разумеется"),
]


def paraphrase_unreferenced(call: dict, rng: random.Random) -> dict | None:
    """Синонимические замены только в репликах без фактов."""
    c = copy.deepcopy(call)
    used = {f["source"]["turn_idx"] for f in c["facts"]}
    changed = False
    for i, t in enumerate(c["turns"]):
        if i in used:
            continue
        new = t["text"]
        for pat, rep in SYNONYM:
            if pat.search(new) and rng.random() < 0.6:
                new = pat.sub(rep, new, count=1)
                changed = True
        t["text"] = new
    if not changed:
        return None
    _reground(c)
    return c


def asr_noise(call: dict, p: float, seed: str) -> dict | None:
    """Шум cfe/asr_noise; span/quote сбрасываются (как у ~45% фактов gold)."""
    from asr_noise import corrupt_call
    c = corrupt_call(call, p, seed)
    c["noise_level"] = round((c.get("noise_level") or 0) + p, 2)
    for f in c["facts"]:
        f["source"]["span"] = None
        f["source"]["quote"] = None
    return c


TRANSFORMS = ["date_shift", "renumber", "tariff_rename", "filler_insert",
              "paraphrase", "asr_noise"]
