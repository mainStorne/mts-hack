"""A mock of the operator's back office: customers, tariffs, outages and tickets, in Postgres (db/schema.sql). The knowledge base
is a table too (db/kb.py).

Replace this module with calls to your real systems; the agent's tools (agent/tools.py) are its only users. The tables are
seeded from data/seed.json by `seed()` (`support-db init` and the start of the API): only keys that are missing are added.
`reset()` makes them exactly the seed again (`support-db reset-demo`, the tests).
"""

import json
import re
import uuid
from dataclasses import dataclass
from importlib import resources

from support.db import pg


def _seed_data() -> dict:
    return json.loads(resources.files("support").joinpath("data", "seed.json").read_text(encoding="utf-8"))


def seed() -> None:
    """Add what data/seed.json has and the tables do not; nothing existing is changed."""
    data = _seed_data()
    customers = [{k: v for k, v in c.items() if k != "charges"} | {"tariff_id": c["tariff"]} for c in data["customers"]]
    charges = [{**ch, "phone": c["phone"]} for c in data["customers"] for ch in c["charges"]]
    with pg.connection() as conn, conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO tariffs (id, name, price, data_gb, minutes, sms, description)
               VALUES (%(id)s, %(name)s, %(price)s, %(data_gb)s, %(minutes)s, %(sms)s, %(description)s) ON CONFLICT DO NOTHING""",
            data["tariffs"],
        )
        cur.executemany(
            """INSERT INTO customers (phone, pin, name, region, tariff_id, balance)
               VALUES (%(phone)s, %(pin)s, %(name)s, %(region)s, %(tariff_id)s, %(balance)s) ON CONFLICT DO NOTHING""",
            customers,
        )
        cur.executemany(
            """INSERT INTO charges (id, phone, date, description, amount)
               VALUES (%(id)s, %(phone)s, %(date)s, %(description)s, %(amount)s) ON CONFLICT DO NOTHING""",
            charges,
        )
        cur.executemany(
            "INSERT INTO outages (region, service, status) VALUES (%(region)s, %(service)s, %(status)s) ON CONFLICT DO NOTHING",
            data["outages"],
        )


def reset() -> None:
    """The back office exactly as data/seed.json has it: every change and every ticket is gone."""
    with pg.connection() as conn:
        conn.execute("TRUNCATE tickets, outages, charges, customers, tariffs")
    seed()


def normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 11 and digits[0] == "8":
        digits = "7" + digits[1:]
    return "+" + digits


TARIFF = "SELECT id, name, price, data_gb, minutes, sms, description FROM tariffs"


def tariffs() -> list[dict]:
    with pg.connection() as conn:
        return conn.execute(TARIFF + " ORDER BY price, id").fetchall()


def tariff(tariff_id: str) -> dict | None:
    with pg.connection() as conn:
        return conn.execute(TARIFF + " WHERE id = %s", (tariff_id,)).fetchone()


def customer(phone: str) -> dict:
    """The customer with their `tariff` (an id) and `charges`, by the normalized phone number."""
    with pg.connection() as conn:
        found = conn.execute(
            "SELECT phone, name, region, tariff_id AS tariff, balance::float8 AS balance FROM customers WHERE phone = %s",
            (phone,),
        ).fetchone()
        if found is None:
            raise KeyError(phone)
        found["charges"] = conn.execute(
            """SELECT id, to_char(date, 'YYYY-MM-DD') AS date, description, amount::float8 AS amount, disputed
               FROM charges WHERE phone = %s ORDER BY date, id""",
            (phone,),
        ).fetchall()
        return found


def verify(phone: str, pin: str) -> dict | None:
    """The customer when the phone and the PIN match, else None."""
    number = normalize_phone(phone)
    with pg.connection() as conn:
        ok = conn.execute("SELECT 1 FROM customers WHERE phone = %s AND pin = %s", (number, pin.strip())).fetchone()
    return customer(number) if ok else None


def change_tariff(phone: str, tariff_id: str) -> None:
    with pg.connection() as conn:
        conn.execute("UPDATE customers SET tariff_id = %s WHERE phone = %s", (tariff_id, phone))


def dispute_charge(phone: str, charge_id: str, reason: str) -> bool:
    with pg.connection() as conn:
        return (
            conn.execute(
                "UPDATE charges SET disputed = %s WHERE id = %s AND phone = %s", (reason, charge_id, phone)
            ).rowcount
            > 0
        )


def outages(region: str) -> list[dict]:
    with pg.connection() as conn:
        return conn.execute(
            "SELECT region, service, status FROM outages WHERE lower(region) = lower(%s) ORDER BY service",
            (region.strip(),),
        ).fetchall()


@dataclass
class Ticket:
    id: str
    customer: str | None
    summary: str
    priority: str
    created_at: str
    notes: list[str]


_TICKET = "SELECT id, phone AS customer, summary, priority, notes, created_at::text AS created_at FROM tickets"


def create_ticket(customer_phone: str | None, summary: str, priority: str = "normal") -> Ticket:
    ticket_id = f"T-{uuid.uuid4().hex[:6].upper()}"
    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO tickets (id, phone, summary, priority) VALUES (%s, %s, %s, %s)",
            (ticket_id, customer_phone, summary, priority),
        )
    return ticket(ticket_id)  # type: ignore[return-value]  # just inserted


def add_ticket_note(ticket_id: str, note: str) -> bool:
    with pg.connection() as conn:
        return conn.execute("UPDATE tickets SET notes = notes || %s WHERE id = %s", ([note], ticket_id)).rowcount > 0


def ticket(ticket_id: str) -> Ticket | None:
    with pg.connection() as conn:
        row = conn.execute(_TICKET + " WHERE id = %s", (ticket_id,)).fetchone()
    return Ticket(**row) if row else None
