"""Database setup.

support-db init    # apply db/schema.sql (safe to repeat; the API also does it when it starts)
"""

import argparse
from pathlib import Path

from support.db import pg

SCHEMA = Path(__file__).resolve().parents[3] / "db" / "schema.sql"


def init() -> None:
    with pg.connection() as conn:
        conn.execute(SCHEMA.read_text(encoding="utf-8"))  # type: ignore[arg-type]  # a file in this repository, not user input


def main() -> None:
    parser = argparse.ArgumentParser(prog="support-db")
    parser.add_argument("command", choices=["init"])
    parser.parse_args()
    init()
    print("schema applied")
    pg.close()


if __name__ == "__main__":
    main()
