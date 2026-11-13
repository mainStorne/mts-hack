"""Knowledge-base search over the Postgres table kb_articles (db/kb.py): full-text and vector search together, then the reranker.

    query ─┬► Postgres full-text search (Russian stemming): `kb_candidates` articles that contain its words ─┐
           └► embeddings ─► pgvector: the `kb_candidates` nearest articles ──────────────────────────────────┤
                                                       reciprocal rank fusion (an article high in either list wins) ◄┘
          ─► reranker (a cross-encoder reads the query and the article together) ─► those with a score >= `reranker_min_score`

Words find exact terms (a tariff name, "APN", "0890") that vectors blur; vectors find paraphrases that share no words. There is
no fallback: EMBEDDINGS_ROUTE is required, and a failing embeddings or reranker server is an error of the tool (the agent is
told and says so). Vectors are 1024-dimensional (the column is vector(1024), db/schema.sql). Embeddings of articles are made by
`sync()`: when the database is set up, and before a search for articles that were added without one.
"""

import httpx

from support.db import kb
from support.kb_import.inputs import INPUTS
from support.settings import settings

_http = httpx.Client(timeout=60)


def enabled() -> bool:
    return settings.embeddings_route is not None


def _post(route, path: str, payload: dict) -> dict:
    response = _http.post(f"{str(route).rstrip('/')}{path}", json=payload)
    response.raise_for_status()
    return response.json()


def embed(texts: list[str], *, query: bool = False) -> list[list[float]]:
    """Vectors for a batch of texts, in the same order."""
    prefix = settings.embeddings_query_prefix if query else ""
    data = _post(
        settings.embeddings_route,
        "/embeddings",
        {"input": [prefix + (t.strip() or " ") for t in texts], "model": settings.embeddings_model},
    )["data"]
    data = sorted(data, key=lambda item: item["index"])
    if [item["index"] for item in data] != list(range(len(texts))):
        raise ValueError("Embeddings response must contain exactly one vector per input")
    return [item["embedding"] for item in data]


def rerank(query: str, documents: list[str]) -> list[float]:
    """The relevance of every document to the query (raw scores), in the same order."""
    results = _post(
        settings.reranker_route,
        "/rerank",
        {"model": settings.reranker_model, "query": query, "documents": documents, "raw_scores": True},
    )["results"]
    results = sorted(results, key=lambda item: item["index"])
    if [item["index"] for item in results] != list(range(len(documents))):
        raise ValueError("Reranker response must score every document exactly once")
    return [item["relevance_score"] for item in results]


def _text(article: dict) -> str:
    return f"{article['title']}\n{article['text']}"[: settings.kb_embed_chars]


def sync() -> int:
    """Embed the articles that have no vector yet (nothing without EMBEDDINGS_ROUTE); how many were embedded."""
    if not enabled():
        return 0
    missing = kb.without_vectors()
    if missing:
        kb.set_vectors(dict(zip((a["id"] for a in missing), embed([_text(a) for a in missing]), strict=True)))
    return len(missing)


RRF_K = 60  # the usual constant of reciprocal rank fusion


def fuse(*rankings: list[dict]) -> list[dict]:
    """One list from several rankings: each article scores 1/(RRF_K + rank) in every list it is in; the best sum first."""
    scores: dict[str, float] = {}
    articles: dict[str, dict] = {}
    for ranking in rankings:
        for rank, article in enumerate(ranking, start=1):
            articles[article["id"]] = article
            scores[article["id"]] = scores.get(article["id"], 0.0) + 1 / (RRF_K + rank)
    return [articles[i] for i in sorted(scores, key=lambda i: -scores[i])]


def _result(article: dict, score: float | None = None) -> dict:
    """What the agent sees of an article: the link, the capped text and, when the operation needs any, the customer's data."""
    result = {
        "id": article["id"],
        "title": article["title"],
        "url": article.get("url"),
        "text": article["text"][: settings.kb_result_chars],
    }
    if score is not None:
        result["score"] = round(score, 3)
    needed = [INPUTS[i] for i in article.get("required_inputs") or [] if i in INPUTS]
    if needed:
        result["required_inputs"] = needed
    return result


def search(query: str) -> list[dict]:
    """The articles that answer the query, best first; empty when none is good enough."""
    if not enabled():
        raise RuntimeError("EMBEDDINGS_ROUTE is not set: the knowledge base is searched by words and vectors together")
    sync()
    (wanted,) = embed([query], query=True)
    candidates = fuse(kb.lexical(query, settings.kb_candidates), kb.nearest(wanted, settings.kb_candidates))
    candidates = candidates[: settings.kb_candidates]
    if settings.reranker_route is None or not candidates:
        return [_result(a) for a in candidates[: settings.kb_results]]
    scores = rerank(query, [_text(a) for a in candidates])
    best = sorted(zip(scores, candidates, strict=True), key=lambda pair: -pair[0])
    return [_result(a, s) for s, a in best[: settings.kb_results] if s >= settings.reranker_min_score]
