"""Knowledge-base search over the Postgres table kb_articles (db/kb.py): vectors find candidates, the reranker orders them and
drops weak matches.

    query ─► embeddings ─► pgvector: the `kb_candidates` nearest articles ─► reranker (a cross-encoder reads the query and
          the article together) ─► those with a score >= `reranker_min_score`

Without EMBEDDINGS_ROUTE (or when its server fails) it is Postgres full-text search. Embeddings of articles are made by
`sync()`: when the database is set up, and before a search for articles that were added without one.
"""

import logging

import httpx

from support.db import kb
from support.settings import settings

log = logging.getLogger("support.retrieval")

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
    return f"{article['title']}\n{article['text']}"


def sync() -> int:
    """Embed the articles that have no vector yet (nothing without EMBEDDINGS_ROUTE); how many were embedded."""
    if not enabled():
        return 0
    missing = kb.without_vectors()
    if missing:
        kb.set_vectors(dict(zip((a["id"] for a in missing), embed([_text(a) for a in missing]), strict=True)))
    return len(missing)


def search(query: str) -> list[dict]:
    """The articles that answer the query, best first; empty when none is good enough."""
    if not enabled():
        return kb.lexical(query, settings.kb_results)
    try:
        sync()
        (wanted,) = embed([query], query=True)
        candidates = kb.nearest(wanted, settings.kb_candidates)
        if settings.reranker_route is None or not candidates:
            return candidates[: settings.kb_results]
        scores = rerank(query, [_text(a) for a in candidates])
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("vector search failed (%s): using full-text search", type(exc).__name__)
        return kb.lexical(query, settings.kb_results)
    best = sorted(zip(scores, candidates, strict=True), key=lambda pair: -pair[0])
    return [{**a, "score": round(s, 3)} for s, a in best[: settings.kb_results] if s >= settings.reranker_min_score]
