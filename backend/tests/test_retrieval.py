"""Knowledge-base search over Postgres with pgvector (TEST_DATABASE_URL must be a database with the `vector` extension)."""

import json
import os

import httpx
import pytest

DATABASE = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE, reason="set TEST_DATABASE_URL to run the knowledge-base tests")

if DATABASE:
    from support import retrieval
    from support.db import init, kb, pg
    from support.settings import settings

TOPICS = ["sim", "роуминг", "интернет"]


def server(request: httpx.Request) -> httpx.Response:
    """Embeddings: a one-hot vector per topic word a text mentions. Reranker: 5 for a text with the query's first word, else -5."""
    body = json.loads(request.content)
    if request.url.path.endswith("/embeddings"):
        vectors = [
            [float(t in text.lower()) + 0.01 for t in TOPICS] + [0.0] * (1024 - len(TOPICS)) for text in body["input"]
        ]
        return httpx.Response(200, json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]})
    word = body["query"].lower().split()[0]
    scores = [5.0 if word in d.lower() else -5.0 for d in body["documents"]]
    return httpx.Response(200, json={"results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]})


@pytest.fixture
def base(monkeypatch):
    monkeypatch.setattr(settings, "database_url", DATABASE)
    monkeypatch.setattr(settings, "embeddings_route", None)  # never the servers of the developer's .env
    monkeypatch.setattr(settings, "reranker_route", None)
    with pg.connection() as conn:
        conn.execute("DROP TABLE IF EXISTS kb_articles")
    init.init()  # the schema and the starting articles
    yield
    pg.close()


@pytest.fixture
def servers(monkeypatch, base):
    monkeypatch.setattr(settings, "embeddings_route", "http://emb/v1")
    monkeypatch.setattr(settings, "reranker_route", "http://rank/v1")
    monkeypatch.setattr(retrieval, "_http", httpx.Client(transport=httpx.MockTransport(server)))


def test_the_starting_articles_are_seeded_once_and_never_overwritten(base):
    with pg.connection() as conn:
        conn.execute("UPDATE kb_articles SET text = 'правка' WHERE id = 'kb-sim-lost'")
    assert kb.seed() == 0
    with pg.connection() as conn:
        assert pg.scalar(conn, "SELECT text FROM kb_articles WHERE id = 'kb-sim-lost'") == "правка"


def test_vectors_and_the_reranker_find_articles_and_drop_weak_matches(servers):
    found = retrieval.search("интернет пропал")
    assert (
        found
        and all("интернет" in (a["title"] + a["text"]).lower() for a in found)
        and all(a["score"] >= 0 for a in found)
    )
    assert retrieval.search("погода завтра") == []  # nothing is good enough: the agent is told so
    with pg.connection() as conn:
        assert pg.scalar(conn, "SELECT count(*) FROM kb_articles WHERE embedding IS NULL") == 0


def test_an_article_added_later_is_embedded_before_the_search(servers):
    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO kb_articles (id, title, text) VALUES ('kb-new', 'Роуминг в Китае', 'Пакет роуминг для Китая')"
        )
    assert "kb-new" in [a["id"] for a in retrieval.search("роуминг Китай")]


def test_the_vector_index_exists(base):
    with pg.connection() as conn:
        assert "hnsw" in pg.scalar(
            conn, "SELECT indexdef FROM pg_indexes WHERE indexname = 'kb_articles_embedding_idx'"
        )


def test_search_without_an_embeddings_server_is_an_error(base):
    with pytest.raises(RuntimeError, match="EMBEDDINGS_ROUTE"):
        retrieval.search("потеряна SIM-карта")


def test_a_failing_embeddings_server_is_an_error_not_a_silent_fallback(base, monkeypatch):
    monkeypatch.setattr(settings, "embeddings_route", "http://emb/v1")
    monkeypatch.setattr(retrieval, "_http", httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))))
    with pytest.raises(httpx.HTTPStatusError):
        retrieval.search("потеряна SIM-карта")
