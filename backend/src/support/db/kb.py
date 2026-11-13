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
            """SELECT id, title, text, url, required_inputs FROM kb_articles WHERE embedding IS NOT NULL
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
            """SELECT id, title, text, url, required_inputs FROM kb_articles, to_tsquery('russian', %s) AS q
               WHERE to_tsvector('russian', title || ' ' || text) @@ q
               ORDER BY ts_rank(to_tsvector('russian', title || ' ' || text), q) DESC LIMIT %s""",
            (" | ".join(words), limit),
        ).fetchall()


_SITE_COLUMNS = (
    "id",
    "source",
    "kind",
    "url",
    "product",
    "section",
    "parent_id",
    "popular",
    "title",
    "text",
    "modified_at",
)


def upsert_site(rows: list[dict]) -> int:
    """Insert or update crawled articles. A changed title or text drops the embedding and the required inputs, so only those
    are redone."""
    if not rows:
        return 0
    columns = ", ".join(_SITE_COLUMNS)
    values = ", ".join(f"%({c})s" for c in _SITE_COLUMNS)
    changed = "kb_articles.text IS DISTINCT FROM EXCLUDED.text OR kb_articles.title IS DISTINCT FROM EXCLUDED.title"
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _SITE_COLUMNS if c != "id")
    with pg.connection() as conn, conn.cursor() as cur:
        cur.executemany(
            f"""INSERT INTO kb_articles ({columns}) VALUES ({values})
                ON CONFLICT (id) DO UPDATE SET {updates}, updated_at = now(),
                  embedding = CASE WHEN {changed} THEN NULL ELSE kb_articles.embedding END,
                  required_inputs = CASE WHEN {changed} THEN NULL ELSE kb_articles.required_inputs END""",
            rows,
        )
        return cur.rowcount


def delete_site_except(source: str, keep: set[str]) -> int:
    with pg.connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM kb_articles WHERE source = %s AND NOT (id = ANY(%s))", (source, list(keep)))
        return cur.rowcount


def site_rows(source: str) -> list[dict]:
    """Embedded rows of a source (articles and faq), with the vector parsed into a list."""
    with pg.connection() as conn:
        rows = conn.execute(
            """SELECT id, kind, title, text, product, section, popular, required_inputs, embedding::text AS embedding
               FROM kb_articles WHERE source = %s AND embedding IS NOT NULL ORDER BY id""",
            (source,),
        ).fetchall()
    return [{**r, "embedding": json.loads(r["embedding"])} for r in rows]


def set_required_inputs(values: dict[str, list[str]]) -> None:
    with pg.connection() as conn, conn.cursor() as cur:
        cur.executemany(
            "UPDATE kb_articles SET required_inputs = %s WHERE id = %s", [(v, i) for i, v in values.items()]
        )


def _literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"
