"""A mock of the operator's back office: customers, tariffs, outages, tickets and the knowledge base, in memory.

Replace this module with calls to your real systems; the agent's tools (agent/tools.py) are its only users. State lives in
the process: a restart resets it to data/seed.json (the conversations themselves are in Postgres).
"""

import copy
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources

_lock = threading.Lock()


def _load(name: str):
    return json.loads(resources.files("support").joinpath("data", name).read_text(encoding="utf-8"))


_seed = _load("seed.json")
_knowledge_base: list[dict] = _load("knowledge_base.json")
TARIFFS: dict[str, dict] = {t["id"]: t for t in _seed["tariffs"]}
_customers: dict[str, dict] = {}
_outages: list[dict] = []
_tickets: dict[str, "Ticket"] = {}


@dataclass
class Ticket:
    id: str
    customer: str | None
    summary: str
    priority: str
    created_at: str
    notes: list[str] = field(default_factory=list)


def reset() -> None:
    """Back to the seed data (tests, and the start of the process)."""
    with _lock:
        _customers.clear()
        _customers.update({c["phone"]: copy.deepcopy(c) for c in _seed["customers"]})
        _outages[:] = copy.deepcopy(_seed["outages"])
        _tickets.clear()


reset()


def normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 11 and digits[0] == "8":
        digits = "7" + digits[1:]
    return "+" + digits


def verify(phone: str, pin: str) -> dict | None:
    """The customer (a copy) when the phone and the PIN match, else None."""
    with _lock:
        customer = _customers.get(normalize_phone(phone))
        return copy.deepcopy(customer) if customer and customer["pin"] == pin.strip() else None


def customer(phone: str) -> dict:
    with _lock:
        return copy.deepcopy(_customers[phone])


def change_tariff(phone: str, tariff_id: str) -> dict:
    with _lock:
        _customers[phone]["tariff"] = tariff_id
        return copy.deepcopy(_customers[phone])


def dispute_charge(phone: str, charge_id: str, reason: str) -> bool:
    with _lock:
        for charge in _customers[phone]["charges"]:
            if charge["id"] == charge_id:
                charge["disputed"] = reason
                return True
    return False


def outages(region: str) -> list[dict]:
    with _lock:
        wanted = region.strip().lower()
        return [copy.deepcopy(o) for o in _outages if o["region"].lower() == wanted]


def knowledge_base() -> list[dict]:
    return _knowledge_base


def search_knowledge_base(query: str, limit: int = 3) -> list[dict]:
    """Articles ranked by how many words of the query they contain (a stand-in for a real search)."""
    words = {w for w in re.findall(r"\w+", query.lower()) if len(w) > 2}
    scored = []
    for article in _knowledge_base:
        haystack = f"{article['title']} {article['text']}".lower()
        score = sum(1 for w in words if w[: max(4, len(w) - 2)] in haystack)
        if score:
            scored.append((score, article))
    scored.sort(key=lambda item: -item[0])
    return [a for _, a in scored[:limit]]


def create_ticket(customer_phone: str | None, summary: str, priority: str = "normal") -> Ticket:
    with _lock:
        ticket = Ticket(
            id=f"T-{uuid.uuid4().hex[:6].upper()}",
            customer=customer_phone,
            summary=summary,
            priority=priority,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        _tickets[ticket.id] = ticket
        return ticket


def add_ticket_note(ticket_id: str, note: str) -> bool:
    with _lock:
        ticket = _tickets.get(ticket_id)
        if ticket is None:
            return False
        ticket.notes.append(note)
        return True


def ticket(ticket_id: str) -> Ticket | None:
    with _lock:
        return _tickets.get(ticket_id)
