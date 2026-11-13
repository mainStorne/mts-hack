"""Knowledge-base search: vectors find candidates, the reranker orders them and drops weak matches.

    query ─► embeddings ─► cosine against the articles' vectors (embedded once, kept in the process) ─► top `kb_candidates`
          ─► reranker (a cross-encoder reads query and article together) ─► those with a score >= `reranker_min_score`

Without EMBEDDINGS_ROUTE it is crm.search_knowledge_base (word match). With a real knowledge base (hundreds of articles and
more) keep the vectors in Postgres with pgvector instead of in memory; the docker-compose image already has it.
"""

import logging
import math
import threading

import httpx

from support import crm
from support.settings import settings

log = logging.getLogger("support.retrieval")

_http = httpx.Client(timeout=60)
_lock = threading.Lock()
_vectors: dict[str, list[float]] = {}


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


def _article_vectors() -> dict[str, list[float]]:
    with _lock:
        missing = [a for a in crm.knowledge_base() if a["id"] not in _vectors]
        if missing:
            _vectors.update(zip((a["id"] for a in missing), embed([_text(a) for a in missing]), strict=True))
        return dict(_vectors)


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def search(query: str) -> list[dict]:
    """The articles that answer the query, best first; empty when none is good enough."""
    if not enabled():
        return crm.search_knowledge_base(query, settings.kb_results)
    try:
        vectors = _article_vectors()
        (wanted,) = embed([query], query=True)
        ranked = sorted(crm.knowledge_base(), key=lambda a: -_cosine(wanted, vectors[a["id"]]))[
            : settings.kb_candidates
        ]
        if settings.reranker_route is None:
            return ranked[: settings.kb_results]
        scores = rerank(query, [_text(a) for a in ranked])
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        log.warning("vector search failed (%s): falling back to word match", type(exc).__name__)
        return crm.search_knowledge_base(query, settings.kb_results)
    best = sorted(zip(scores, ranked, strict=True), key=lambda pair: -pair[0])
    return [{**a, "score": round(s, 3)} for s, a in best[: settings.kb_results] if s >= settings.reranker_min_score]
