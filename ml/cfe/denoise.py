#!/usr/bin/env python3
"""Нормализатор ASR-шума: исправление слов по частотному словарю train (правки на расстоянии 1). Только stdlib."""
import copy
import functools
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clf_task  # noqa: E402

ROOT = clf_task.ROOT
WORD_RE = re.compile(r"[а-яё]+", re.I)
ALPHA = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
MIN_LEN = 4
RATIO = 20
MIN_FREQ = 3
VOCAB_PATH = os.path.join(ROOT, "data", "denoise_vocab.json")
_VOCAB = None


def load_vocab():
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError("data/denoise_vocab.json: run python3 cfe/denoise.py")
    with open(VOCAB_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _edits1(w):
    sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    out = {a + b[1:] for a, b in sp if b}
    out |= {a + c + b[1:] for a, b in sp if b for c in ALPHA}
    out |= {a + c + b for a, b in sp for c in ALPHA}
    return out


@functools.lru_cache(maxsize=None)
def fix_word(w):
    global _VOCAB
    if len(w) < MIN_LEN:
        return w
    if _VOCAB is None:
        _VOCAB = load_vocab()
    lw = w.lower()
    need = max(MIN_FREQ, RATIO * max(_VOCAB.get(lw, 0), 1))
    best, bf = None, 0
    for e in _edits1(lw):
        f = _VOCAB.get(e, 0)
        if f >= need and (f > bf or (f == bf and e < best)):
            best, bf = e, f
    if best is None:
        return w
    return best[:1].upper() + best[1:] if w[:1].isupper() else best


def denoise_text(text):
    """-> (чистый текст, to_orig длины len(clean)+1: индекс в чистом тексте -> индекс в исходном)."""
    out, to_orig, pos = [], [], 0
    for m in WORD_RE.finditer(text):
        s, e = m.span()
        for i in range(pos, s):
            out.append(text[i])
            to_orig.append(i)
        new = fix_word(m.group(0))
        for j, ch in enumerate(new):
            out.append(ch)
            to_orig.append(s + min(j, e - s - 1))
        pos = e
    for i in range(pos, len(text)):
        out.append(text[i])
        to_orig.append(i)
    to_orig.append(len(text))
    return "".join(out), to_orig


def denoise_call(call):
    c = copy.deepcopy(call)
    maps = []
    for t in c["turns"]:
        t["text"], m = denoise_text(t["text"])
        maps.append(m)
    return c, maps


def restore_sources(facts, call, maps):
    """Спаны фактов с чистого текста -> на исходный текст звонка `call`."""
    for f in facts:
        src = f["source"]
        if src.get("span"):
            t = src["turn_idx"]
            s, e = maps[t][src["span"][0]], maps[t][src["span"][1]]
            src["span"] = [s, e]
            src["quote"] = call["turns"][t]["text"][s:e]
    return facts


def build_vocab():
    from collections import Counter
    cnt = Counter()
    for c in clf_task.load_calls(clf_task.TRAIN_PATH):
        for t in c["turns"]:
            cnt.update(w.lower() for w in WORD_RE.findall(t["text"]))
        for f in c["facts"]:
            cnt.update({w.lower(): 5 for w in WORD_RE.findall(f["value"])})
    cat = clf_task.load_catalog()
    for v in cat.values():
        for s in v:
            if isinstance(s, str):
                cnt.update({w.lower(): 50 for w in WORD_RE.findall(s)})
    return dict(cnt)


if __name__ == "__main__":
    v = build_vocab()
    with open(VOCAB_PATH, "w", encoding="utf-8") as fh:
        json.dump(v, fh, ensure_ascii=False)
    print(f"vocab={len(v)}")
