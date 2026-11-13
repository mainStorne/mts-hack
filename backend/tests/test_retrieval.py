import httpx
import pytest

from support import retrieval
from support.settings import settings

TOPICS = ["sim", "роуминг", "интернет"]


def server(request: httpx.Request) -> httpx.Response:
    """Embeddings: a one-hot vector per topic word the text mentions. Reranker: 5 for a text with the query's first word, else -5."""
    import json

    body = json.loads(request.content)
    if request.url.path.endswith("/embeddings"):
        vectors = [[float(t in text.lower()) for t in TOPICS] for text in body["input"]]
        return httpx.Response(200, json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]})
    word = body["query"].lower().split()[0]
    return httpx.Response(
        200,
        json={
            "results": [
                {"index": i, "relevance_score": 5.0 if word in d.lower() else -5.0}
                for i, d in enumerate(body["documents"])
            ]
        },
    )


@pytest.fixture
def servers(monkeypatch):
    monkeypatch.setattr(settings, "embeddings_route", "http://emb/v1")
    monkeypatch.setattr(settings, "reranker_route", "http://rank/v1")
    monkeypatch.setattr(retrieval, "_http", httpx.Client(transport=httpx.MockTransport(server)))
    retrieval._vectors.clear()


def test_the_reranker_drops_weak_matches(servers):
    found = retrieval.search("интернет пропал")
    assert (
        found
        and all("интернет" in (a["title"] + a["text"]).lower() for a in found)
        and all(a["score"] >= 0 for a in found)
    )
    assert retrieval.search("погода завтра") == []  # nothing is good enough: the agent is told so


def test_a_failing_server_falls_back_to_word_match(monkeypatch):
    monkeypatch.setattr(settings, "embeddings_route", "http://emb/v1")
    monkeypatch.setattr(retrieval, "_http", httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))))
    retrieval._vectors.clear()
    assert retrieval.search("потеряна SIM-карта")[0]["id"] == "kb-sim-lost"
