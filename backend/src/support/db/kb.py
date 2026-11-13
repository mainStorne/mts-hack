"""The knowledge base in Postgres, table kb_articles (db/schema.sql). Searching it is retrieval.py's job; this module is SQL.

Vectors go to and from Postgres as text ("[0.1,0.2,...]" cast to `vector`), so no extra client library is needed.
"""

import json
import re
from importlib import resources

from support.db import pg

SEED = "knowledge_base.json"


def seed() -> int:
    """Insert the articles of data/knowledge_base.json that the table does not have yet; how many were added. Existing
    articles are never overwritten: the table is the source of truth."""
    articles = json.loads(resources.files("support").joinpath("data", SEED).read_text(encoding="utf-8"))
    with pg.connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO kb_articles (id, title, text) VALUES (%(id)s, %(title)s, %(text)s) ON CONFLICT (id) DO NOTHING",
            articles,
        )
        return max(cur.rowcount, 0)


def without_vectors() -> list[dict]:
    """Articles that have no embedding yet: {id, title, text}."""
    with pg.connection() as conn:
        return conn.execute("SELECT id, title, text FROM kb_articles WHERE embedding IS NULL ORDER BY id").fetchall()


def set_vectors(vectors: dict[str, list[float]]) -> None:
    with pg.connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE kb_articles SET embedding = %s::vector WHERE id = %s",
            [(_literal(v), article_id) for article_id, v in vectors.items()],
        )


def nearest(vector: list[float], limit: int) -> list[dict]:
    """The articles closest to the vector by cosine distance, nearest first (articles without a vector are not searched)."""
    with pg.connection() as conn:
        return conn.execute(
            """SELECT id, title, text FROM kb_articles WHERE embedding IS NOT NULL
               ORDER BY embedding <=> %s::vector LIMIT %s""",
            (_literal(vector), limit),
        ).fetchall()


def lexical(query: str, limit: int) -> list[dict]:
    """Full-text search (Russian stemming): articles that contain any word of the query, the best matching first."""
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    if not words:
        return []
    with pg.connection() as conn:
        return conn.execute(
            """SELECT id, title, text FROM kb_articles, to_tsquery('russian', %s) AS q
               WHERE to_tsvector('russian', title || ' ' || text) @@ q
               ORDER BY ts_rank(to_tsvector('russian', title || ' ' || text), q) DESC LIMIT %s""",
            (" | ".join(words), limit),
        ).fetchall()


def _literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"
