"""Детерминированная нормализация относительных дат («вчера», «со вторника», «через два дня») к дате звонка."""
import re
from datetime import date, datetime, timedelta

WD_GEN = {"понедельника": 0, "вторника": 1, "среды": 2, "четверга": 3, "пятницы": 4, "субботы": 5, "воскресенья": 6}
WD_ACC = {"понедельник": 0, "вторник": 1, "среду": 2, "четверг": 3, "пятницу": 4, "субботу": 5, "воскресенье": 6}
NUM = {"один": 1, "одного": 1, "два": 2, "две": 2, "двух": 2, "три": 3, "трёх": 3, "трех": 3, "четыре": 4, "четырёх": 4,
       "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "пару": 2, "пары": 2}
NUMRX = r"(\d{1,2}|" + "|".join(NUM) + r")"
ADJ = {"позавчерашн": "позавчера", "вчерашн": "вчера", "сегодняшн": "сегодня", "завтрашн": "завтра", "послезавтрашн": "послезавтра"}
FIXED = {"позавчера": -2, "вчера": -1, "сегодня": 0, "завтра": 1, "послезавтра": 2}

PATTERNS = [
    ("days_ago", re.compile(rf"\b{NUMRX}\s+дн(?:ей|я|ю)?\s+(?:тому\s+)?назад")),
    ("days_ago", re.compile(rf"\bдн(?:ей|я|ю)?\s+{NUMRX}\s+назад")),
    ("in_days", re.compile(rf"\bчерез\s+{NUMRX}\s+дн")),
    ("from_wd", re.compile(r"(?:^|\W)(?:с|со)\s+(" + "|".join(WD_GEN) + r")\b")),
    ("on_wd", re.compile(r"(?:^|\W)(?:в|во|до|к|на)\s+(" + "|".join(WD_ACC) + r")\b")),
    ("on_wd_gen", re.compile(r"(?:^|\W)(?:до|после|с|со)\s+(" + "|".join(WD_GEN) + r")\b")),
    ("fixed", re.compile(r"\b(позавчера|вчера|сегодня|завтра|послезавтра)\b")),
    ("fixed_adj", re.compile(r"\b(позавчерашн|вчерашн|сегодняшн|завтрашн|послезавтрашн)\w*")),
    ("week_ago", re.compile(r"\bнеделю\s+назад")),
    ("week_ahead", re.compile(r"\bчерез\s+неделю")),
    ("last_week", re.compile(r"\bна\s+прошл\w*\s+недел\w*")),
    ("month_ago", re.compile(r"\bмесяц\s+назад")),
    ("last_month", re.compile(r"\bв\s+прошлом\s+месяце")),
    ("two_weeks", re.compile(r"\bчерез\s+две\s+недели")),
    ("next_week", re.compile(r"\bна\s+следующей\s+неделе")),
    ("these_days", re.compile(r"\bна\s+днях")),
]


def _num(s):
    return int(s) if s.isdigit() else NUM.get(s)


def resolve(text, call_dt: datetime, future_bias=False):
    """→ [(kind, date, char_pos)] — все относительные даты в тексте."""
    base = call_dt.date(); low = text.lower(); out = []
    for kind, rx in PATTERNS:
        for m in rx.finditer(low):
            g = m.group(1) if m.groups() else None
            d = None
            if kind == "days_ago" and _num(g): d = base - timedelta(days=_num(g))
            elif kind == "in_days" and _num(g): d = base + timedelta(days=_num(g))
            elif kind == "from_wd":
                wd = WD_GEN[g]; d = base - timedelta(days=(base.weekday() - wd) % 7 or 7)
            elif kind in ("on_wd", "on_wd_gen"):
                wd = WD_ACC.get(g, WD_GEN.get(g))
                past = base - timedelta(days=(base.weekday() - wd) % 7 or 7)
                fut = base + timedelta(days=(wd - base.weekday()) % 7 or 7)
                d = ("wd", past, fut)
            elif kind == "fixed": d = base + timedelta(days=FIXED[g])
            elif kind == "fixed_adj": d = base + timedelta(days=FIXED[ADJ[g]])
            elif kind == "week_ago": d = base - timedelta(days=7)
            elif kind == "week_ahead": d = base + timedelta(days=7)
            elif kind == "last_week": d = base - timedelta(days=base.weekday() + 7)       # понедельник прошлой недели
            elif kind == "month_ago": d = base - timedelta(days=31)
            elif kind == "last_month": d = (base.replace(day=1) - timedelta(days=1)).replace(day=1)
            elif kind == "two_weeks": d = base + timedelta(days=14)
            elif kind == "next_week": d = base + timedelta(days=7 - base.weekday())
            elif kind == "these_days": d = base - timedelta(days=3)
            if d is not None:
                out.append((kind, d, m.start()))
    return sorted(out, key=lambda x: x[2])


def pick(dates, base: date, future: bool):
    """выбор одной даты из найденных; для дней недели — прошлое или будущее по смыслу факта"""
    for kind, d, pos in dates:
        if isinstance(d, tuple):
            return d[2] if future else d[1]
        if future and d >= base:
            return d
        if not future and d <= base:
            return d
    return None


def iso(d):
    return d.isoformat() + "T00:00:00" if isinstance(d, date) and not isinstance(d, datetime) else d
