"""Построитель синтетического звонка: сущности, реплики, факты.

Реплики собираются из сегментов: свободный текст ("f") и цитаты ("q", с ключом).
Каждый факт привязывается к реплике-носителю (carrier) и, при желании, к цитатному
сегменту. Смещения дат задаются относительно started_at и конвертируются в ISO
после сборки. Итоговая структура полностью совпадает со схемой кейса.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from . import catalog as cat
from .dates import DateGen, iso
from .noise import LEVELS, render_turn

ATTRS = {"service_state", "tariff", "ticket_status", "action", "agreement"}


class Obj:
    def __init__(self, oid: str, otype: str, name: str):
        self.id, self.type, self.name = oid, otype, name

    def as_dict(self):
        return {"id": self.id, "type": self.type, "name": self.name}


class Handle:
    """Ссылка на реплику-носитель с цитатными сегментами по ключам."""

    def __init__(self, idx: int):
        self.idx = idx
        self.quotes: dict[str, dict] = {}  # key -> {"text", "span", "protected"}


class Builder:
    def __init__(self, rng: random.Random, *, account_seed: int = 0):
        self.rng = rng
        self.account_id = f"u9{rng.randint(1, 39):02d}_{rng.randint(10000, 99999)}"
        if account_seed:
            self.account_id += f"s{account_seed}"
        self.call_serial = rng.randint(1, 2999)
        self.call_id = f"{self.account_id}:call{self.call_serial:04d}"
        day = date(2026, 1, 5) + timedelta(days=rng.randint(0, 285))
        hour = rng.randint(8, 19)
        self.started_at = f"{day.isoformat()}T{hour:02d}:00:00"
        self.duration = None
        self.noise_level = None
        self.turns: list[dict] = []
        self.facts: list[dict] = []
        self._handles: dict[int, Handle] = {}
        self._svc_n = 0
        self._dev_n = 0
        self._tkt_n = 0
        self.dg = DateGen(self.started_at, rng)
        first = rng.choice(cat.FIRST_NAMES)
        middle = rng.choice(cat.MIDDLE_NAMES)
        self.client_obj = Obj(f"{self.account_id}:client", "client", f"{first} {middle}")
        self.series = None

    # ---------------------------------------------------------------- объекты
    def service(self, name: str | None = None) -> Obj:
        if name is None:
            name = self.rng.choice(list(cat.SERVICE_PROFILES))
        oid = f"{self.account_id}:svc{self._svc_n}"
        self._svc_n += 1
        return Obj(oid, "service", name)

    def profile(self, svc: Obj) -> dict:
        return cat.SERVICE_PROFILES.get(svc.name, {"colloq": [svc.name], "symptoms": []})

    def device(self, name: str | None = None) -> Obj:
        name = name or self.rng.choice(cat.DEVICES)
        oid = f"{self.account_id}:dev{self._dev_n}"
        self._dev_n += 1
        return Obj(oid, "device", name)

    def ticket(self, num: int | None = None) -> Obj:
        num = num if num is not None else self.rng.randint(3000, 99999)
        oid = f"{self.account_id}:ticket{self._tkt_n}"
        self._tkt_n += 1
        return Obj(oid, "ticket", f"заявка {num}")

    @property
    def client(self) -> Obj:
        return self.client_obj

    # ---------------------------------------------------------------- реплики
    def carrier(self, speaker: str, parts: list) -> Handle:
        """parts: ("f", text) | ("q", text, key) | "текст" (свободный).
        Голые строки допустимы для удобства, но явные ("f", ...) предпочтительнее."""
        idx = len(self.turns)
        h = Handle(idx)
        normalized = []
        for p in parts:
            if isinstance(p, str):
                p = ("f", p)
            if p[0] == "q":
                normalized.append(("q", p[1]))
                h.quotes[p[2]] = {"key": p[2], "pending": True}
            else:
                normalized.append(("f", p[1]))
        self.turns.append({"speaker": speaker, "parts": normalized})
        self._handles[idx] = h
        return h

    def turn(self, speaker: str, *chunks: str) -> int:
        """Обычная реплика без цитат."""
        idx = len(self.turns)
        self.turns.append({"speaker": speaker, "parts": [("f", c) for c in chunks]})
        return idx

    def append_free(self, handle: Handle, text: str):
        """Добавить свободный текст в конец реплики-носителя."""
        self.turns[handle.idx]["parts"].append(("f", text))

    def add_quote(self, handle: Handle, text: str, key: str):
        """Добавить новый цитатный сегмент в реплику-носитель."""
        self.turns[handle.idx]["parts"].append(("q", text))
        handle.quotes[key] = {"key": key, "pending": True}

    def last_speaker(self) -> str | None:
        return self.turns[-1]["speaker"] if self.turns else None

    # ---------------------------------------------------------------- факты
    def close_last(self, obj, vf_off: int, vt_off: int):
        """Закрыть последний открытый single-valued факт объекта obj с днём
        valid_from == vf_off (вчера закрыли, сегодня новое состояние)."""
        from datetime import timedelta
        day = (date.fromisoformat(self.started_at[:10]) + timedelta(days=vf_off)).isoformat()
        for f in reversed(self.facts):
            if (f["object"]["id"] == obj.id
                    and f["attribute"] in {"service_state", "tariff", "ticket_status"}
                    and f["valid_to"] is None
                    and f["valid_from"][:10] == day):
                f["valid_to"] = iso(self.started_at, vt_off, None)
                return
        raise KeyError(f"нечего закрывать: {obj.id} vf={vf_off}")

    def fact(self, handle: Handle, obj: Obj, attribute: str, value: str,
             *, vf: int, vt: int | None = None, hour: int | None = None,
             polarity: int = 1, quote_key: str | None = None,
             span: bool = True, quote: bool = True):
        """Зарегистрировать факт на реплике handle.

        vf/vt — смещения в днях от дня звонка. hour — озвученное время суток
        для valid_from. quote_key — сегмент-цитата в этой реплике.
        span/quote решают, попадут ли эти поля в source (имитация золота).
        """
        if attribute not in ATTRS:
            raise ValueError(attribute)
        turn_idx = handle.idx
        turn = self.turns[turn_idx]
        if not quote:
            span = False
        if span and not quote:
            raise ValueError("span без quote невозможен: span ищется по quote")
        want_text = quote  # quote нужен и для span
        src_quote, src_span = None, None
        if want_text and quote_key is not None:
            if quote_key not in handle.quotes:
                raise KeyError(f"нет цитатного сегмента {quote_key!r}")
            self._pending_quote(handle, quote_key)
            src_quote = quote_key  # разрешается после наложения шума
            if span:
                src_span = quote_key
        elif want_text and quote_key is None:
            raise ValueError("quote=True требует quote_key")
        self.facts.append({
            "_turn_idx": turn_idx, "_speaker": turn["speaker"],
            "_quote_key": src_quote, "_span_key": src_span,
            "object": obj.as_dict(), "attribute": attribute, "value": value,
            "polarity": polarity,
            "valid_from": iso(self.started_at, vf, hour),
            "valid_to": None if vt is None else iso(self.started_at, vt, None),
        })
        return self.facts[-1]

    def fact_on_turn(self, idx: int, obj: Obj, attribute: str, value: str, **kw):
        """Факт на обычной реплике (turn), созданной через b.turn(...).
        Цитаты у такой реплики нет, поэтому quote=False (как в gold:
        «помехи длятся 5 дней и не прошли», quote NULL)."""
        if kw.get("quote", True):
            raise ValueError("у реплики из b.turn() нет цитатных сегментов: используйте quote=False")
        h = Handle(idx)
        return self.fact(h, obj, attribute, value, **kw)

    def _pending_quote(self, handle: Handle, key: str):
        handle.quotes[key]["pending"] = True

    # ---------------------------------------------------------------- сборка
    def finalize(self, noise: str = "moderate") -> dict:
        p_free, p_quote, level = LEVELS[noise]
        turns_out, quote_res = [], {}
        for i, t in enumerate(self.turns):
            text, quotes = render_turn(t["parts"], p_free, p_quote, self.rng, add_disfl=True)
            turns_out.append({"idx": i, "speaker": t["speaker"], "text": text})
            quote_res[i] = quotes
        facts_out = []
        for f in self.facts:
            src = {"turn_idx": f["_turn_idx"], "span": None, "quote": None}
            qk, sk = f["_quote_key"], f["_span_key"]
            if qk is not None:
                # порядок цитат в turn.parts соответствует порядку в quote_res
                handle = self._handles[f["_turn_idx"]]
                pos = list(handle.quotes).index(qk)
                seg = quote_res[f["_turn_idx"]][pos]
                src["quote"] = seg["text"]
                if sk is not None:
                    src["span"] = seg["span"]
            facts_out.append({
                "object": f["object"], "attribute": f["attribute"], "value": f["value"],
                "polarity": f["polarity"], "speaker": f["_speaker"],
                "valid_from": f["valid_from"], "valid_to": f["valid_to"],
                "source": src, "confirmed": True,
            })
        call = {
            "call_id": self.call_id,
            "started_at": self.started_at,
            "duration_sec": 60 + 24 * sum(len(t["text"]) for t in turns_out) // 40,
            "series": self.series,
            "turns": turns_out,
            "noise_level": level,
            "facts": facts_out,
        }
        self._quote_res = quote_res
        return call
