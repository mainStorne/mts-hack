import json
from pathlib import Path

from support.kb_import.clusters import aggregate_inputs, cluster, neighbors
from support.kb_import.inputs import INPUTS
from support.kb_import.mts import article_rows

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "mts_article.json").read_text(encoding="utf-8"))
URL = "https://support.mts.ru/mts_mobilnyy_internet/prodlenie-interneta/chto-delat-yesli-zakonchilsya-paket-interneta"


def test_article_and_its_faq_become_rows():
    rows = article_rows(FIXTURE, URL, {81846})
    assert [r["id"] for r in rows] == [
        "mts-81846",
        "mts-81846-faq-kak-podklyuchit-bezlimitnyy-internet",
        "mts-81846-faq-kak-uvelichit-paket-interneta",
    ]
    article, faq = rows[0], rows[1]
    assert "Единоразово" in article["text"]
    assert "Часто спрашивают" not in article["text"]
    assert "Как увеличить пакет интернета в настройках тарифа" not in article["text"]
    assert faq["title"] == "Как подключить безлимитный интернет"
    assert faq["parent_id"] == "mts-81846" and faq["kind"] == "faq"
    assert article["popular"] is True and article["product"] == "Мобильный интернет"


def test_cluster_is_deterministic():
    vectors = [[1, 0.1], [1, 0.0], [0.9, 0.1], [0.1, 1], [0.0, 1], [0.1, 0.9]]
    labels, centroids = cluster(vectors, 2)
    assert len(centroids) == 2
    assert labels[0] == labels[1] == labels[2] != labels[3] == labels[4] == labels[5]
    assert cluster(vectors, 2)[0] == labels


def test_aggregate_inputs():
    got = aggregate_inputs([["passport", "phone"], ["passport"], ["sim"], []], 0.25)
    assert got == [
        {"id": "passport", "label": INPUTS["passport"], "share": 0.5},
        {"id": "phone", "label": INPUTS["phone"], "share": 0.25},
        {"id": "sim", "label": INPUTS["sim"], "share": 0.25},
    ]


def test_neighbors():
    assert neighbors([[1, 0], [0.9, 0.1], [0, 1]], 1) == [[1], [0], [1]]
