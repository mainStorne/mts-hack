"""Нормализация текста и поиск записей каталога в репликах (лемматизация pymorphy3, нечёткие совпадения)."""
import json, re
from difflib import SequenceMatcher
from functools import lru_cache

import pymorphy3

_M = pymorphy3.MorphAnalyzer()
WORD = re.compile(r"[а-яёa-z0-9]+(?:-[а-яёa-z0-9]+)*")
TRANSLIT = {"wifi": "вай-фай", "wi-fi": "вай-фай", "вайфай": "вай-фай", "вай-фаю": "вай-фай", "tv": "тв",
            "lan": "лан", "sms": "смс", "sim": "сим", "ip": "ип", "gpon": "гпон", "ont": "онт"}
# разговорные варианты → каталожная лексика
SYN = {"инет": "интернет", "интернетик": "интернет", "телек": "телевизор", "телик": "телевизор",
       "симка": "сим-карта", "комп": "компьютер", "смска": "смс", "эсэмэска": "смс", "роутор": "роутер",
       "модемчик": "модем", "приставочка": "приставка", "мастер": "специалист", "техник": "специалист",
       "инженер": "специалист", "монтажник": "специалист"}
STOP = {"в", "на", "по", "для", "с", "со", "о", "об", "за", "к", "от", "до", "и", "а", "у", "из", "при"}


@lru_cache(maxsize=300_000)
def lemma(w: str) -> str:
    w = w.lower().replace("ё", "е")
    w = TRANSLIT.get(w, w)
    if "-" in w:
        parts = [lemma(p) for p in w.split("-") if p]
        return "-".join(parts)
    lw = _M.parse(w)[0].normal_form.replace("ё", "е")
    return SYN.get(lw, lw)


def tokens(text: str):
    """[(lemma, start, end)]"""
    out = []
    for m in WORD.finditer(text.lower()):
        w = m.group()
        if w in TRANSLIT:
            w = TRANSLIT[w]
        out.append((lemma(w), m.start(), m.end()))
    return out


def stem(w):
    return w[:-2] if len(w) >= 7 else w[:-1] if len(w) >= 5 else w


@lru_cache(maxsize=500_000)
def close(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4:
        return False
    if stem(a) == stem(b):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.84


class Catalog:
    SECTION_ATTR = {"symptoms": "service_state", "actions": "action", "agreements": "agreement",
                    "tariffs": "tariff", "statuses": "ticket_status"}

    def __init__(self, path):
        self.raw = json.load(open(path, encoding="utf-8"))
        self.entries = []   # (section, value, [content lemmas])
        for sec, vals in self.raw.items():
            for v in vals:
                lem = [l for l, _, _ in tokens(v) if l not in STOP]
                if lem:
                    self.entries.append((sec, v, lem))
        self.by_first = {}
        for i, (sec, v, lem) in enumerate(self.entries):
            for l in set(lem):
                self.by_first.setdefault(l[:3], []).append(i)

    def find(self, toks, sections=None, window_extra=3):
        """Ищет записи каталога в списке токенов. → [(entry_idx, score, start_char, end_char, tok_i0, tok_i1)]"""
        lem = [t[0] for t in toks]
        cand = set()
        for l in lem:
            cand.update(self.by_first.get(l[:3], ()))
        res = []
        for ei in cand:
            sec, v, E = self.entries[ei]
            if sections and sec not in sections:
                continue
            k = len(E)
            # позиции каждого слова записи
            pos = [[i for i, l in enumerate(lem) if close(e, l)] for e in E]
            present = [p for p in pos if p]
            need = k if k <= 2 else k - 1
            if len(present) < need:
                continue
            # лучшее окно: перебираем якорь по первой найденной позиции
            best = None
            for anchor in sorted({i for p in present for i in p}):
                lo, hi, hit = anchor, anchor, 0
                for p in pos:
                    near = [i for i in p if abs(i - anchor) <= k + window_extra]
                    if near:
                        j = min(near, key=lambda i: abs(i - anchor))
                        lo, hi, hit = min(lo, j), max(hi, j), hit + 1
                if hit >= need and (hi - lo) <= k + window_extra:
                    sc = hit / k
                    if best is None or sc > best[0] or (sc == best[0] and hi - lo < best[2] - best[1]):
                        best = (sc, lo, hi)
            if best:
                sc, lo, hi = best
                res.append((ei, sc, toks[lo][1], toks[hi][2], lo, hi))
        return res
