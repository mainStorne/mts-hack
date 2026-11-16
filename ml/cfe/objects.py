"""Имя объекта (service/device) по упоминанию в реплике: поиск названий каталога по основам слов. Только stdlib + cfe."""
import functools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textsim  # noqa: E402
import clf_task  # noqa: E402

WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.I)
STOP = {"для", "на", "за", "от", "по", "из", "без", "при"}


def _stem(w):
    w = w.lower().replace("ё", "е")
    return w if len(w) <= 4 else w[:max(4, len(w) - 2)]


@functools.lru_cache(maxsize=None)
def _names():
    cat = clf_task.load_catalog()
    out = []
    for kind, key in (("service", "services"), ("device", "devices")):
        for n in cat[key]:
            st = [_stem(w) for w in WORD_RE.findall(n) if w.lower() not in STOP]
            if st:
                out.append((n, kind, st))
    return out


@functools.lru_cache(maxsize=None)
def mentions(text):
    """((позиция, название, тип, длина основ), ...): названия каталога, все слова которых есть в тексте."""
    toks = [(m.start(), _stem(m.group())) for m in WORD_RE.finditer(text)]
    out = []
    for n, kind, st in _names():
        first = None
        for s in st:
            pos = [i for i, x in toks if x.startswith(s) or (len(x) >= 4 and s.startswith(x))]
            if not pos:
                break
            first = min(pos) if first is None else min(first, min(pos))
        else:
            out.append((first, n, kind, len(" ".join(st))))
    return tuple(out)


def object_name(call, t, kind, start):
    """Название объекта типа `kind` для спана, начинающегося в реплике t с позиции `start`:
    ближайшее упоминание в той же реплике, иначе последнее в предыдущих репликах; для service/device,
    если не найдено, то же для другого типа; '' если нет."""
    kinds = [kind]
    if kind in ("service", "device"):
        kinds.append("device" if kind == "service" else "service")
    for k in kinds:
        for u in range(t, -1, -1):
            ms = [m for m in mentions(call["turns"][u]["text"]) if m[2] == k]
            if ms:
                ms.sort(key=lambda m: (abs(m[0] - start) if u == t else -m[0], -m[3]))
                return ms[0][1]
    return ""


def snap_name(text, kind, th=0.5):
    """Ближайшее по textsim._value_sim название каталога типа kind; '' если сходство < th."""
    best, bs = "", 0.0
    for n, k, _ in _names():
        if k == kind:
            s = textsim._value_sim(text, n)
            if s > bs:
                best, bs = n, s
    return best if bs >= th else ""
