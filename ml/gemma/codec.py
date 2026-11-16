"""Промпт, цель и декодер для CompactFactExtraction. Только stdlib + case_metrics."""
import calendar
import json
import re
from datetime import date, timedelta

import sys
from pathlib import Path

# официальный оценщик v2.1 (источник SINGLE_VALUED/_value_sim) лежит в data/raw
sys.path.insert(0, str(Path(__file__).resolve().parent / "data/raw"))
from case_metrics import SINGLE_VALUED, _value_sim  # noqa: E402

WEEKDAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
ATTRS = {"service_state", "tariff", "ticket_status", "action", "agreement"}

SYSTEM = (
    "Ты извлекаешь факты из транскрипта звонка в техподдержку (текст с ошибками распознавания речи). "
    "Верни только JSON-массив фактов без пояснений. Поля факта: t — номер реплики-источника; "
    "o — тип объекта (client|service|device|ticket|other); n — имя объекта; "
    "a — атрибут (service_state|tariff|ticket_status|action|agreement); v — нормализованное значение; "
    "p — 1 или -1 (только явное отрицание клиента); f — смещение даты начала в днях от дня звонка; "
    "e — смещение даты окончания или null. Не являются фактами: неподтверждённые клиентом предложения "
    "оператора, отвергнутые догадки, вопросы, гипотезы, посторонние темы. "
    "Новое значение тарифа, статуса заявки или состояния услуги закрывает старое (e)."
)

CATALOG_CATEGORY = {
    "service_state": "symptoms",
    "tariff": "tariffs",
    "ticket_status": "statuses",
    "action": "actions",
    "agreement": "agreements",
}


def _d(iso: str) -> date:
    return date.fromisoformat(iso[:10])


def _signed(n: int) -> str:
    return f"+{n}" if n > 0 else str(n)


def date_header(started_at: str) -> str:
    d = _d(started_at)
    wd = d.weekday()
    py, pm = (d.year, d.month - 1) if d.month > 1 else (d.year - 1, 12)
    prev = date(py, pm, min(d.day, calendar.monthrange(py, pm)[1]))
    month_back = (prev - d).days
    past = [-(((wd - k) % 7) or 7) for k in range(7)]
    future = [((k - wd) % 7) or 7 for k in range(7)]
    return "\n".join([
        f"Начало звонка: {d.isoformat()} {started_at[11:16]}, {WEEKDAYS[wd]}",
        f"Смещения: вчера -1, позавчера -2, завтра +1, послезавтра +2, неделю назад -7, месяц назад {month_back}",
        "Прошедшие дни недели: " + ", ".join(f"{WEEKDAYS[k]} {past[k]}" for k in range(7))
        + "; ближайшие будущие: " + ", ".join(f"{WEEKDAYS[k]} {_signed(future[k])}" for k in range(7)),
    ])


def render_user(call: dict) -> str:
    lines = [f"Звонок: {call['call_id']}", date_header(call["started_at"]), "Диалог:"]
    for t in call["turns"]:
        lines.append(f"[{t['idx']}] {'К' if t['speaker'] == 'client' else 'О'}: {t['text']}")
    return "\n".join(lines)


def messages(call: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render_user(call)}]


def day_offset(started_at: str, iso: str) -> int:
    return (_d(iso) - _d(started_at)).days


def iso(started_at: str, k: int) -> str:
    if k == 0:
        return started_at
    return (_d(started_at) + timedelta(days=k)).isoformat() + "T00:00:00"


def encode_target(call: dict) -> str:
    s = call["started_at"]
    out = []
    for f in call["facts"]:
        out.append({
            "t": f["source"]["turn_idx"],
            "o": f["object"]["type"],
            "n": f["object"]["name"],
            "a": f["attribute"],
            "v": f["value"],
            "p": f["polarity"],
            "f": day_offset(s, f["valid_from"]),
            "e": None if f.get("valid_to") is None else day_offset(s, f["valid_to"]),
        })
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def _parse(text: str):
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    m = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*(```)?$", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text)
    except ValueError:
        pass
    cut = text.rfind("}")
    if cut < 0:
        return None
    try:
        return json.loads(text[: cut + 1] + "]")
    except ValueError:
        return None


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def decode_target(text: str, call: dict) -> tuple[list[dict], int]:
    """(факты, число отброшенных); второй элемент -1 — вывод не разобран."""
    arr = _parse(text)
    if not isinstance(arr, list):
        return [], -1
    s = call["started_at"]
    turns = call["turns"]
    facts, dropped = [], 0
    for x in arr:
        if not (
            isinstance(x, dict)
            and x.get("a") in ATTRS
            and x.get("p") in (1, -1) and _is_int(x.get("p"))
            and _is_int(x.get("t")) and 0 <= x["t"] < len(turns)
            and _is_int(x.get("f"))
            and (x.get("e") is None or _is_int(x.get("e")))
            and isinstance(x.get("v"), str)
            and isinstance(x.get("n"), str)
            and isinstance(x.get("o"), str)
        ):
            dropped += 1
            continue
        try:
            vf = iso(s, x["f"])
            vt = None if x["e"] is None else iso(s, x["e"])
        except (OverflowError, ValueError):
            dropped += 1
            continue
        facts.append({
            "object": {"type": x["o"], "name": x["n"]},
            "attribute": x["a"],
            "value": x["v"].strip(),
            "polarity": x["p"],
            "speaker": turns[x["t"]]["speaker"],
            "valid_from": vf,
            "valid_to": vt,
            "source": {"turn_idx": x["t"], "span": None, "quote": None},
        })
    return facts, dropped


def close_states(facts: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for f in facts:
        if f["attribute"] in SINGLE_VALUED:
            groups.setdefault((f["object"]["name"].lower(), f["attribute"]), []).append(f)
    for g in groups.values():
        g.sort(key=lambda f: f["valid_from"])
        for cur, nxt in zip(g, g[1:]):
            if cur["valid_to"] is None:
                cur["valid_to"] = nxt["valid_from"]
    return facts


def snap_to_catalog(facts: list[dict], catalog: dict) -> list[dict]:
    for f in facts:
        cands = catalog.get(CATALOG_CATEGORY[f["attribute"]], [])
        if f["value"] in cands:
            continue
        best, score = None, 0.0
        for c in cands:
            sc = _value_sim(f["value"], c)
            if sc > score:
                best, score = c, sc
        if best is not None and score >= 0.9:
            f["value"] = best
    return facts
