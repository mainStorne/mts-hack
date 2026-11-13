"""Analysis of the crawled articles: which customer data each operation needs (stored with the row) and groups of articles by
embeddings (KMeans), printed as a report for people deciding where a specialist agent with its own tools is worth adding."""

import asyncio
import logging
from collections import Counter
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from support.agent.core import _model
from support.db import kb
from support.kb_import.inputs import INPUTS
from support.kb_import.mts import SOURCE

log = logging.getLogger(__name__)


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def cluster(vectors: list[list[float]], k: int) -> tuple[list[int], list[list[float]]]:
    """KMeans over L2-normalised vectors: (a label per vector, normalised centroids)."""
    x = _normalise(np.array(vectors, dtype=float))
    model = KMeans(n_clusters=min(k, len(x)), n_init=10, random_state=0).fit(x)
    return model.labels_.tolist(), _normalise(model.cluster_centers_).tolist()


def neighbors(centroids: list[list[float]], n: int) -> list[list[int]]:
    """For each centroid the indices of the n other centroids with the highest dot product, highest first."""
    c = np.array(centroids, dtype=float)
    sims = c @ c.T
    result = []
    for i in range(len(c)):
        order = [j for j in np.argsort(-sims[i], kind="stable") if j != i]
        result.append([int(j) for j in order[:n]])
    return result


def aggregate_inputs(lists: list[list[str]], min_share: float, limit: int = 5) -> list[dict]:
    """The inputs that at least min_share of the articles need, the most common first."""
    if not lists:
        return []
    order = list(INPUTS)
    shares = {i: sum(i in lst for lst in lists) / len(lists) for i in INPUTS}
    chosen = sorted(
        (i for i in order if shares[i] >= min_share and shares[i] > 0), key=lambda i: (-shares[i], order.index(i))
    )
    return [{"id": i, "label": INPUTS[i], "share": round(shares[i], 2)} for i in chosen[:limit]]


def report(clusters: list[dict], articles: list[dict]) -> str:
    if not clusters:
        return "нет кластеров"
    out: list[str] = []
    for i, c in enumerate(clusters):
        card = c["card"]
        out += [
            f"## Кластер {i}: {card.label} ({c['size']} статей)",
            "",
            f"Описание: {card.description}",
            "",
            "Разделы:",
        ]
        out += [f"- {name}: {count}" for name, count in c["sections"].most_common()]
        out += ["", "Примеры:"] + [f"- {t}" for t in c["examples"]]
        out += ["", "Чаще всего нужно клиенту:"]
        out += [f"- {r['label']} — {r['share']:.0%}" for r in c["required_inputs"]] or ["- ничего особенного"]
        out += [
            "",
            "Ближайшие кластеры: " + ", ".join(f"{j} ({clusters[j]['card'].label})" for j in c["neighbors"]),
            "",
        ]
    analysed = [a for a in articles if a.get("required_inputs") is not None]
    out += ["## Данные клиента по всей базе", "", "| данные | доля статей |", "| --- | --- |"]
    shares = {
        i: (sum(i in a["required_inputs"] for a in analysed) / len(analysed) if analysed else 0.0) for i in INPUTS
    }
    out += [f"| {INPUTS[i]} | {shares[i]:.0%} |" for i in sorted(INPUTS, key=lambda i: -shares[i])]
    return "\n".join(out)


class _Needs(BaseModel):
    inputs: list[Literal[tuple(INPUTS)]]  # type: ignore[valid-type]


class TopicCard(BaseModel):
    label: str = Field(max_length=40)
    description: str = Field(max_length=200)


def _llm():
    return _model(max_tokens=300, temperature=0, extra_body={"chat_template_kwargs": {"enable_thinking": False}})


async def analyse(articles: list[dict], parallel: int) -> dict[str, list[str]]:
    """Required inputs of the articles that have none yet. A failure stores [] so it is not retried forever."""
    todo = [a for a in articles if a.get("required_inputs") is None]
    vocabulary = "\n".join(f"{i}: {label}" for i, label in INPUTS.items())
    model = _llm().with_structured_output(_Needs, method="json_schema")
    gate = asyncio.Semaphore(parallel)
    found: dict[str, list[str]] = {}

    async def one(a: dict) -> None:
        prompt = (
            "The text is an instruction from a mobile operator's help centre. Which of these does the customer need to "
            "provide, have or do to complete the operation it describes? Choose only what the text requires; an empty "
            f"list is fine.\n\n{vocabulary}\n\nTitle: {a['title']}\n\n{a['text'][:4000]}"
        )
        async with gate:
            try:
                needs = await model.ainvoke(prompt)
                found[a["id"]] = list(dict.fromkeys(needs.inputs))
            except Exception as e:  # noqa: BLE001
                log.warning("%s: %s", a["id"], e)
                found[a["id"]] = []

    await asyncio.gather(*(one(a) for a in todo))
    return found


async def name_cluster(products: list[str], sections: list[str], titles: list[str]) -> TopicCard:
    model = _llm().with_structured_output(TopicCard, method="json_schema")
    prompt = (
        "These help-centre articles were grouped together. Give the support specialist who handles them a short Russian "
        "name (2–5 words, no quotes) and one Russian sentence describing what customers ask them about.\n\n"
        f"Products: {'; '.join(products)}\nSections: {'; '.join(sections)}\nTitles:\n" + "\n".join(titles[:15])
    )
    return await model.ainvoke(prompt)


async def analyse_site(*, k: int, min_share: float, min_size: int, parallel: int, refresh_inputs: bool) -> str:
    rows = kb.site_rows(SOURCE)
    if not rows:
        raise RuntimeError("no embedded support.mts.ru articles: run `support-kb crawl` with EMBEDDINGS_ROUTE set")
    if refresh_inputs:
        for r in rows:
            r["required_inputs"] = None
    found = await analyse(rows, parallel)
    if found:
        kb.set_required_inputs(found)
        for r in rows:
            if r["id"] in found:
                r["required_inputs"] = found[r["id"]]
    articles = [r for r in rows if r["kind"] == "article"]
    labels, centroids = cluster([a["embedding"] for a in articles], k)
    groups: dict[int, list[dict]] = {}
    for a, label in zip(articles, labels, strict=True):
        groups.setdefault(label, []).append(a)
    kept = sorted((g for g in groups.items() if len(g[1]) >= min_size), key=lambda g: -len(g[1]))
    if not kept:
        return report([], articles)
    near = neighbors([centroids[label] for label, _ in kept], 2)
    clusters = []
    for (_, members), nb in zip(kept, near, strict=True):
        titles = [a["title"] for a in sorted(members, key=lambda a: (not a["popular"], a["title"]))]
        sections = Counter(f"{a['product']} / {a['section']}" for a in members)
        clusters.append(
            {
                "size": len(members),
                "examples": titles[:8],
                "sections": sections,
                "required_inputs": aggregate_inputs([a["required_inputs"] or [] for a in members], min_share),
                "neighbors": nb,
                "card": await name_cluster(
                    sorted({a["product"] for a in members}), sorted({a["section"] for a in members}), titles
                ),
            }
        )
    return report(clusters, articles)
