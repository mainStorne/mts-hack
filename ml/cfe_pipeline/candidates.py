"""Генерация кандидатов фактов из звонка: упоминания записей каталога, статусы заявок, тарифы."""
import re
from textnorm import Catalog, tokens, lemma

STATUS_RE = [
    ("назначен выезд специалиста", re.compile(r"(назначен\w*\s+(?:\w+\s+){0,2}выезд|выезд\w*\s+(?:\w+\s+){0,3}назначен|статус\w*\s+(?:\w+\s+){0,2}«?назначен)")),
    ("выполнена", re.compile(r"\b(выполнен\w*|закрыт[аао]?\b|исполнен\w*)")),
    ("в работе", re.compile(r"\bв\s+работ[уеы]\b|\bв\s+работе\b|«в работе»|взяли\s+в\s+работу|передан\w*\s+в\s+работу")),
    ("принята", re.compile(r"\b(принят[аоы]?\b|принял\w*|зарегистрир\w*|оформлен[аоы]?\b|оформил\w*|заведен\w*|создан[аоы]?\b|создал\w*)")),
]
TICKET_NUM = re.compile(r"заяв\w*\s*(?:номер\s*|№\s*|за\s+номером\s+)?(\d{3,7})|номер\w*\s+заявк\w*\s*(\d{3,7})|(?:№|номер)\s*(\d{3,7})")
TARIFF_CTX = re.compile(r"тариф|«|»|пакет|план")


def ticket_numbers(text):
    return [next(g for g in m.groups() if g) for m in TICKET_NUM.finditer(text.lower())]


class CandidateGen:
    def __init__(self, catalog_path):
        self.cat = Catalog(catalog_path)

    def call(self, d):
        turns = d["turns"]
        toks = [tokens(t["text"]) for t in turns]
        # объекты: услуги/устройства, номера заявок
        objs = []   # (turn, kind, name)
        tickets = []  # (turn, num)
        for i, t in enumerate(turns):
            for ei, sc, s, e, a, b in self.cat.find(toks[i], sections={"services", "devices"}, window_extra=1):
                if sc >= 1.0:
                    sec, v, _ = self.cat.entries[ei]
                    objs.append((i, "service" if sec == "services" else "device", v))
            for n in ticket_numbers(t["text"]):
                tickets.append((i, n))
        cands = []
        for i, t in enumerate(turns):
            low = t["text"].lower()
            for ei, sc, s, e, a, b in self.cat.find(toks[i], sections={"symptoms", "actions", "agreements", "tariffs"}):
                sec, v, lem = self.cat.entries[ei]
                attr = Catalog.SECTION_ATTR[sec]
                if attr == "tariff":
                    raw = t["text"][s:e]
                    ctx = low[max(0, s - 30):e + 5]
                    if not (raw[:1].isupper() or TARIFF_CTX.search(ctx)):
                        continue
                    if len(lem) == 1 and sc < 1:
                        continue
                cands.append(dict(turn=i, attr=attr, value=v, score=sc, start=s, end=e, tok0=a, tok1=b, entry=ei))
            if re.search(r"заяв|статус|обращени", low) or (i and re.search(r"заяв", turns[i - 1]["text"].lower())):
                for val, rx in STATUS_RE:
                    for m in rx.finditer(low):
                        cands.append(dict(turn=i, attr="ticket_status", value=val, score=1.0, start=m.start(),
                                          end=m.end(), tok0=None, tok1=None, entry=-1))
        return cands, objs, tickets, toks
