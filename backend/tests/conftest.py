"""Every test works on a real Postgres with pgvector (TEST_DATABASE_URL), the same as the app: the back office, the knowledge
base and the chats are tables. Without it the tests are skipped."""

import os

import pytest

DATABASE = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(autouse=True)
def fresh_database(monkeypatch):
    if not DATABASE:
        pytest.skip("set TEST_DATABASE_URL to run the tests (Postgres with the pgvector extension)")
    from support import crm
    from support.db import init, pg
    from support.settings import settings

    monkeypatch.setattr(settings, "database_url", DATABASE)
    monkeypatch.setattr(settings, "embeddings_route", None)  # never the servers of the developer's .env
    monkeypatch.setattr(settings, "reranker_route", None)
    init.init()
    crm.reset()
    yield
    pg.close()
