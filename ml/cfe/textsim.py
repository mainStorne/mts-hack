"""Похожесть строк прежнего оценщика (v1): snap, locate_value, имена объектов, правила дат. Оценка — case_metrics.py (официальный v2.1)."""
import re
from difflib import SequenceMatcher

SINGLE_VALUED = {"service_state", "tariff", "ticket_status"}

WORD_RE = re.compile(r"[а-яёa-z0-9]+")


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower().replace("ё", "е"))


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _value_sim(a: str, b: str) -> float:
    """Похожесть значений: точная строка → пословенно-жадная (порядок слов важен)."""
    if a.strip().lower() == b.strip().lower():
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return _sim(a.lower(), b.lower())
    if ta == tb:
        return 1.0
    # жадное выравнивание слов в порядке следования
    j, hits = 0, 0
    for w in ta:
        best, bj = 0.0, None
        for k in range(j, len(tb)):
            s = _sim(w, tb[k])
            if s > best:
                best, bj = s, k
        if best >= 0.75:
            hits += 1
            j = (bj or j) + 1
    return hits / max(len(ta), len(tb))
