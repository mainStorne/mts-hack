"""Сегментный ASR-шум, портированный из cfe/asr_noise.py.

Ключевое отличие: реплика собирается из сегментов «свободных» и «цитатных».
Цитатные сегменты (носители фактов) искажаются сильно слабее и только
буквенными подстановками — цифры, даты и номера заявок не портятся,
отрицания не удаляются. span/quote пересчитываются по финальному тексту,
поэтому разметка остаётся точной при любом уровне шума.
"""
from __future__ import annotations

import random

CONFUSE = {"а": "о", "о": "а", "е": "и", "и": "е", "ы": "и", "з": "с", "с": "з",
           "б": "п", "п": "б", "д": "т", "т": "д", "в": "ф", "ф": "в",
           "ж": "ш", "ш": "ж", "г": "к", "к": "г"}
ALPHABET = "абвгдежзиклмнопрстуфхцчшщыэюя"
DISFLUENCY = ["ээ", "ну", "вот", "эээ", "как бы"]

LEVELS = {
    # имя: (p_свободный_текст, p_цитата, уровень поля noise_level)
    "clean": (0.0, 0.0, 0.0),
    "light": (0.07, 0.02, 0.15),
    "moderate": (0.20, 0.06, 0.25),
}


def _maybe_word(w: str) -> bool:
    return bool(w) and any(ch.isalpha() for ch in w)


def corrupt_word(w: str, p: float, rng: random.Random, safe: bool) -> str:
    if not _maybe_word(w) or len(w) < 3:
        return w
    if rng.random() >= p:
        return w
    i = rng.randrange(len(w))
    ch = w[i]
    if ch.isdigit() or ch in "«»()—-.,!?…":
        return w
    if not safe:
        # удаление буквы (как в оригинальном asr_noise, op<0.8)
        if rng.random() < 0.2:
            return w[:i] + w[i + 1:]
    if ch.lower() in CONFUSE and rng.random() < 0.6:
        rep = CONFUSE[ch.lower()]
        return w[:i] + (rep.upper() if ch.isupper() else rep) + w[i + 1:]
    if not safe and rng.random() < 0.25:
        return w[:i] + rng.choice(ALPHABET) + w[i + 1:]
    return w


def corrupt_segment(text: str, p: float, rng: random.Random, safe: bool,
                    allow_drop_word: bool, allow_truncate: bool,
                    protect_tail: bool = False) -> str:
    words = text.split(" ")
    out = []
    n = len(words)
    for i, w in enumerate(words):
        # последнее слово сегмента не выбрасываем: реплика не должна обрываться
        # посреди фразы («...не знаю, что и» вместо «...что и думать»)
        if (allow_drop_word and i < n - 1 and rng.random() < p / 4
                and n > 3 and _maybe_word(w) and len(w) > 3):
            # не трогаем служебно-критичные слова
            if w.lower().strip(".,!?") not in {"не", "нет", "да"}:
                continue
        out.append(corrupt_word(w, p, rng, safe))
    s = " ".join(out)
    if allow_truncate and not protect_tail and rng.random() < p / 2 and len(s) > 40:
        # обрыв только по границе слова, не посреди слова
        k = int(len(s) * rng.uniform(0.75, 0.92))
        cut = s.rfind(" ", 0, k)
        if cut > 20:
            s = s[:cut] + " —"
    return s


def render_turn(parts: list[tuple], p_free: float, p_quote: float,
                rng: random.Random, add_disfl: bool) -> tuple[str, list[dict]]:
    """Собрать текст реплики из сегментов.

    parts: список ("f", text) | ("q", text)
    Возвращает (финальный текст, список цитат [{text, span}])."""
    chunks: list[str] = []
    quotes: list[dict] = []
    for j, (kind, text) in enumerate(parts):
        is_last = j == len(parts) - 1
        if kind == "f":
            if p_free > 0:
                if add_disfl and rng.random() < p_free and text and text[0].isupper() and not text.startswith("—"):
                    text = rng.choice(DISFLUENCY) + " " + text[0].lower() + text[1:]
                text = corrupt_segment(text, p_free, rng, safe=False,
                                       allow_drop_word=True, allow_truncate=True,
                                       protect_tail=is_last)
            chunks.append(text)
        else:  # цитата
            if p_quote > 0:
                text = corrupt_segment(text, p_quote, rng, safe=True,
                                       allow_drop_word=False, allow_truncate=False)
            start = sum(len(c) for c in chunks)
            chunks.append(text)
            quotes.append({"text": text, "span": [start, start + len(text)]})
    return "".join(chunks), quotes
