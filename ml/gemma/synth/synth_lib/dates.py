"""Движок относительных дат, выровненный по семантике date_header в cfe/codec.py
и проверенный по gold train.jsonl:

  * "вчера" -1, "позавчера" -2, "сегодня" 0, "завтра" +1, "послезавтра" +2;
  * "N дня/дней назад" -N (N=3..6), "через два/три/... дня/дней" +N (N=2..6);
  * "неделю назад" -7, "через неделю" +7;
  * ближайший прошедший день недели ("во вторник", "со вторника", "до пятницы"):
        -(((wd - k) % 7) or 7)   — ровно -7, если сегодня этот день;
  * ближайший будущий день недели ("в среду", "в пятницу"):
        +(((k - wd) % 7) or 7)   — ровно +7, если сегодня этот день;
  * "месяц назад" / "в прошлом месяце": тот же день предыдущего месяца с зажимом,
        как в codec.date_header.
Время суток (час) проставляем только при явно озвученной фразе
("утром" -> 09:00, "к обеду" -> 13:00, "после трёх" -> 15:00 ...).
"""
from __future__ import annotations

import calendar
import random
from datetime import date, timedelta

WD_ACC = ["понедельник", "вторник", "среду", "четверг", "пятницу", "субботу",
          "воскресенье"]
WD_GEN = ["понедельника", "вторника", "среды", "четверга", "пятницы", "субботы",
          "воскресенья"]
NUM_WORDS = {2: "два", 3: "три", 4: "четыре", 5: "пять", 6: "шесть",
             7: "семь", 8: "восемь", 9: "девять", 10: "десять"}
_NUM_FORM = {2: "дня", 3: "дня", 4: "дня", 5: "дней", 6: "дней", 7: "дней",
             8: "дней", 9: "дней", 10: "дней"}

HOURS = {"утром": 9, "с утра": 9, "к обеду": 13, "в обед": 13, "после обеда": 14,
         "после двух": 14, "после трёх": 15, "после четырёх": 16,
         "после пяти": 17, "после шести": 18, "вечером": 19, "после восьми": 20}

# фразы-часы, допустимые для конкретных смещений
_HOUR_ALLOW = {
    0: ["утром", "с утра", "к обеду", "в обед", "после обеда", "после двух",
        "после трёх", "после четырёх", "после пяти", "после шести", "вечером"],
    -1: ["утром", "с утра", "к обеду", "в обед", "после обеда", "вечером",
         "после шести", "после восьми"],
    -2: ["утром", "вечером"],
}


def d0(started_at: str) -> date:
    return date.fromisoformat(started_at[:10])


def month_back(started_at: str) -> int:
    d = d0(started_at)
    py, pm = (d.year, d.month - 1) if d.month > 1 else (d.year - 1, 12)
    prev = date(py, pm, min(d.day, calendar.monthrange(py, pm)[1]))
    return (prev - d).days


def past_weekday(started_at: str, k: int) -> int:
    wd = d0(started_at).weekday()
    return -(((wd - k) % 7) or 7)


def future_weekday(started_at: str, k: int) -> int:
    wd = d0(started_at).weekday()
    return ((k - wd) % 7) or 7


def iso(started_at: str, day_off: int, hour: int | None = None) -> str:
    if day_off == 0 and hour is None:
        return started_at
    day = d0(started_at) + timedelta(days=day_off)
    if hour is None:
        return day.isoformat() + "T00:00:00"
    return day.isoformat() + f"T{hour:02d}:00:00"


class DateGen:
    """Подбор фраз, канонически соответствующих заданному смещению в днях."""

    def __init__(self, started_at: str, rng: random.Random):
        self.started_at = started_at
        self.rng = rng

    # ---- точное смещение -> (фраза, час|None, вариант-описание) -------------
    def month_back(self) -> int:
        return month_back(self.started_at)

    def exact(self, off: int, *, hour: int | None = None, tense: str | None = None):
        """(фраза, озвученный час) для точного day-offset. Час передаётся как
        пожелание; если для смещения нет часовой фразы, вернётся None."""
        r = self.rng
        cands: list[tuple[str, int | None]] = []
        if tense is None:
            tense = "past" if off < 0 else ("future" if off > 0 else "neutral")
        if off == -1:
            cands += [("вчера", None), ("вчера", None), ("вчера", None)]
        elif off == -2:
            cands += [("позавчера", None)]
        elif off == 0:
            cands += [("сегодня", None), ("сегодня", None)]
        elif off == 1:
            cands += [("завтра", None), ("завтра", None)]
        elif off == 2:
            cands += [("послезавтра", None)]
        elif off == -7:
            cands += [("неделю назад", None)]
        elif off == 7:
            cands += [("через неделю", None)]
        elif off == month_back(self.started_at):
            cands += [("месяц назад", None), ("в прошлом месяце", None)]
        elif off < 0 and 3 <= -off <= 10:
            n = -off
            cands += [(f"{NUM_WORDS[n]} {_NUM_FORM[n]} назад", None)]
        elif off > 0 and 2 <= off <= 6:
            n = off
            cands += [(f"через {NUM_WORDS[n]} {_NUM_FORM[n]}", None)]
        # часовые уточнения
        if hour is not None and off in _HOUR_ALLOW:
            for ph, h in HOURS.items():
                if h == hour and ph in _HOUR_ALLOW[off]:
                    cands.append((f"{'сегодня' if off == 0 else 'вчера' if off == -1 else 'позавчера'} {ph}", h))
        if not cands:
            return None, None
        return r.choice(cands)

    # ---- прошедший день недели ----------------------------------------------
    def past_wd(self, k: int, form: str = "acc"):
        off = past_weekday(self.started_at, k)
        if off == 0:
            off = -7  # сегодня этот день недели -> прошлый ровно неделю назад
        if form == "acc":
            pre = "во" if k == 1 else "в"
            ph = f"{pre} {WD_ACC[k]}"
        elif form == "gen_from":
            pre = "со" if k in (1, 2, 5) else "с"
            ph = f"{pre} {WD_GEN[k]}"
        elif form == "gen_to":
            ph = f"до {WD_GEN[k]}"
        else:
            raise ValueError(form)
        return ph, off

    def future_wd(self, k: int):
        off = future_weekday(self.started_at, k)
        if off == 0:
            off = +7
        pre = "во" if k == 1 else "в"
        return f"{pre} {WD_ACC[k]}", off
