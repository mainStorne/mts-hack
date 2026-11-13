"""Database setup.

support-db init    # schema + starting knowledge base + its vectors (safe to repeat; the API also does it when it starts)
"""

import argparse
import logging
from pathlib import Path

from support import retrieval
from support.db import kb, pg

log = logging.getLogger("support.db")

SCHEMA = Path(__file__).resolve().parents[3] / "db" / "schema.sql"


def init() -> None:
    """The schema, the starting knowledge base and the vectors of its articles."""
    with pg.connection() as conn:
        conn.execute(SCHEMA.read_text(encoding="utf-8"))  # type: ignore[arg-type]  # a file in this repository, not user input
    kb.seed()
    try:
        retrieval.sync()
    except Exception as exc:  # noqa: BLE001 - the embeddings server may be down; the search embeds what is missing when it runs
        log.warning("could not embed the knowledge base now (%s)", type(exc).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(prog="support-db")
    parser.add_argument("command", choices=["init"])
    parser.parse_args()
    init()
    print("schema applied")
    pg.close()


if __name__ == "__main__":
    main()
