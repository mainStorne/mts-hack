"""Выбор valid_from / valid_to среди кандидатов-дат (ранжировщик на LogisticRegression)."""
from datetime import datetime
import dates as DT

TEMP_ATTRS = {"service_state", "tariff", "ticket_status", "agreement"}


def date_options(d, c):
    """кандидаты дат для упоминания c: [(iso_day|None, feats)]"""
    call_dt = datetime.fromisoformat(d["started_at"]); base = call_dt.date()
    ti = c["turn"]; a = c["attr"]; v = c["value"]
    opts = []
    common = {f"attr={a}": 1, f"val={a}:{v}": 1}
    found = []
    for off in (-1, 0, 1):
        j = ti + off
        if not (0 <= j < len(d["turns"])):
            continue
        txt = d["turns"][j]["text"]; n = len(DT.resolve(txt, call_dt))
        for kind, x, pos in DT.resolve(txt, call_dt):
            prev = txt[max(0, pos - 12):pos].lower().split()
            prep = prev[-1] if prev else "^"
            if off == 0:
                dist = pos - c["start"]
                db = "same:" + ("before" if dist < 0 else "after") + ":" + str(min(4, abs(dist) // 25))
            else:
                db = f"turn{off:+d}"
            variants = [("past", x[1]), ("fut", x[2])] if isinstance(x, tuple) else [("", x)]
            for var, day in variants:
                rel = "past" if day < base else ("fut" if day > base else "today")
                f = dict(common)
                f.update({f"kind={kind}": 1, f"kind_var={kind}{var}": 1, f"rel={rel}": 1, f"db={db}": 1,
                          f"prep={prep}": 1, f"a_kind={a}:{kind}{var}": 1, f"a_db={a}:{db}": 1,
                          f"a_prep={a}:{prep}": 1, f"a_rel={a}:{rel}": 1, f"v_prep={v}:{prep}": 1,
                          f"v_db={v}:{db}": 1, f"ndates={min(n, 4)}": 1})
                found.append((day.isoformat(), f))
    opts += found
    opts.append((base.isoformat(), {**common, "kind=call": 1, f"a_kind={a}:call": 1, f"v_call={v}": 1,
                                    f"has_dates={int(bool(found))}": 1}))
    return opts


def to_options(d, c):
    """кандидаты valid_to: None + даты"""
    opts = [(None, {f"attr={c['attr']}": 1, f"val={c['attr']}:{c['value']}": 1, "kind=none": 1,
                    f"v_none={c['value']}": 1})]
    for day, f in date_options(d, c):
        if "kind=call" in f:
            f = {**f, "is_callday": 1}
        opts.append((day, f))
    return opts
