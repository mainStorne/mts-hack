"""Дополнительный ASR-шум для аугментации и шумного среза val. Только stdlib."""
import argparse
import copy
import json
import random

CONFUSE = {"а": "о", "о": "а", "е": "и", "и": "е", "ы": "и", "з": "с", "с": "з", "б": "п", "п": "б",
           "д": "т", "т": "д", "в": "ф", "ф": "в", "ж": "ш", "ш": "ж", "г": "к", "к": "г"}
ALPHABET = "абвгдежзиклмнопрстуфхцчшщыэюя"


def corrupt_text(text, p, rng):
    words = text.split(" ")
    out = []
    for w in words:
        r = rng.random()
        if r < p / 4 and len(words) > 3:
            continue
        if r < p and len(w) >= 3:
            i = rng.randrange(len(w))
            ch, op = w[i], rng.random()
            if ch.lower() in CONFUSE and op < 0.6:
                rep = CONFUSE[ch.lower()]
                w = w[:i] + (rep.upper() if ch.isupper() else rep) + w[i + 1:]
            elif op < 0.8:
                w = w[:i] + w[i + 1:]
            else:
                w = w[:i] + rng.choice(ALPHABET) + w[i + 1:]
        out.append(w)
    s = " ".join(out)
    if rng.random() < p / 2 and len(s) > 40:
        s = s[: int(len(s) * rng.uniform(0.6, 0.9))].rstrip() + " —"
    return s


def corrupt_call(call, p, seed):
    rng = random.Random(f"{seed}:{call['call_id']}")
    c = copy.deepcopy(call)
    for t in c["turns"]:
        t["text"] = corrupt_text(t["text"], p, rng)
    return c


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    with open(a.dst, "w", encoding="utf-8") as fh:
        for line in open(a.src, encoding="utf-8"):
            c = corrupt_call(json.loads(line), a.p, a.seed)
            c["noise_level"] = round((c.get("noise_level") or 0) + a.p, 2)
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
