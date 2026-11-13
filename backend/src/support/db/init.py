"""Database setup.

support-db init         # schema + starting knowledge base + its vectors (safe to repeat; the API also does it when it starts)
support-db reset-demo   # the demo customers, tariffs, outages back to data/seed.json; tickets and changes are gone
"""

import argparse
import logging
from pathlib import Path

from support import crm, retrieval
from support.db import kb, pg

log = logging.getLogger("support.db")

SCHEMA = Path(__file__).resolve().parents[3] / "db" / "schema.sql"


def init() -> None:
    """The schema, the starting knowledge base and back office, and the vectors of the articles."""
    with pg.connection() as conn:
        conn.execute(SCHEMA.read_text(encoding="utf-8"))  # type: ignore[arg-type]  # a file in this repository, not user input
    kb.seed()
    crm.seed()
    try:
        retrieval.sync()
    except Exception as exc:  # noqa: BLE001 - the embeddings server may be down; the search embeds what is missing when it runs
        log.warning("could not embed the knowledge base now (%s)", type(exc).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(prog="support-db")
    parser.add_argument("command", choices=["init", "reset-demo"])
    command = parser.parse_args().command
    init()
    if command == "reset-demo":
        crm.reset()  # customers, tariffs, outages and tickets exactly as data/seed.json has them (the knowledge base is kept)
    print("schema applied" if command == "init" else "demo back office reset")
    pg.close()


if __name__ == "__main__":
    main()
