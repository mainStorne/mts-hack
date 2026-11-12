"""The API against a real Postgres (TEST_DATABASE_URL) with a scripted model. Skipped without a database."""

import json
import os

import pytest

DATABASE = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE, reason="set TEST_DATABASE_URL to run the API tests")

if DATABASE:
    from fastapi.testclient import TestClient

    from support import crm
    from support.api import app as app_module
    from support.api import runner
    from support.db import pg
    from support.settings import settings
    from tests.helpers import ScriptedModel, call, say, use

HEADERS = {"X-Client-Id": "browser-one-1234"}
OTHER = {"X-Client-Id": "browser-two-5678"}


@pytest.fixture
def script(monkeypatch):
    """Start the app with an agent whose model says what the test appended to `model.script`."""
    monkeypatch.setattr(settings, "database_url", DATABASE)
    model = ScriptedModel(script=[])
    real = runner.build_agent
    monkeypatch.setattr(
        runner, "build_agent", lambda checkpointer=None: real(model, guardrail=False, checkpointer=checkpointer)
    )
    monkeypatch.setattr(runner.titles, "name", lambda question: "Тест")
    with TestClient(app_module.app) as client:
        client.model = model
        yield client
    with pg.connection() as conn:
        conn.execute("DELETE FROM chats")
    pg.close()


def events(response) -> list[tuple[str, dict]]:
    found, name = [], None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            found.append((name, json.loads(line.removeprefix("data: "))))
    return found


def test_a_handoff_is_streamed_and_kept_in_the_chat(script):
    script.model.script += [use(call("transfer_to_tech", reason="нет интернета")), say("Какой у вас город?")]
    response = script.post("/chats/stream", headers=HEADERS, json={"query": "У меня не работает интернет"})
    got = events(response)
    names = [name for name, _ in got]
    assert names[0] == "chat" and names[-1] == "done"
    assert ("agent", {"name": "tech", "label": "Technical support"}) in got
    done = got[-1][1]
    assert done["answer"] == "Какой у вас город?"
    assert done["state"]["active_agent"] == "tech"
    assert done["handoffs"] == [{"from_agent": "triage", "to_agent": "tech", "reason": "нет интернета"}]

    chat_id = done["chat_id"]
    listed = script.get("/chats", headers=HEADERS).json()
    assert [(c["chat_id"], c["title"]) for c in listed] == [(chat_id, "Тест")]
    history = script.get(f"/chats/{chat_id}", headers=HEADERS).json()
    assert history["state"]["active_agent"] == "tech" and [a["answer_id"] for a in history["answers"]] == [
        done["answer_id"]
    ]

    # The next message of the chat goes to the specialist that was active.
    script.model.script += [say("В Казани идут работы на сети.")]
    second = script.post("/chats/", headers=HEADERS, json={"query": "Казань", "chat_id": chat_id}).json()
    assert second["state"]["active_agent"] == "tech" and second["handoffs"] == []
    assert "Technical-support specialist".lower() in script.model.seen[-1]["system"].lower()


def test_chats_belong_to_their_client(script):
    script.model.script += [say("Здравствуйте!")]
    chat_id = script.post("/chats/", headers=HEADERS, json={"query": "Привет"}).json()["chat_id"]
    assert script.get("/chats", headers=OTHER).json() == []
    assert script.get(f"/chats/{chat_id}", headers=OTHER).status_code == 404
    assert script.post("/chats/", headers=OTHER, json={"query": "Привет", "chat_id": chat_id}).status_code == 404
    assert script.delete(f"/chats/{chat_id}", headers=OTHER).status_code == 404
    assert script.get("/chats").status_code == 401


def test_the_tariff_change_waits_for_the_decision(script):
    # The state of a verified customer in the tariffs department is set by the agent's own steps.
    script.model.script += [
        use(call("transfer_to_tariffs", reason="хочет сменить тариф")),
        use(call("verify_customer", phone="8 999 000-11-22", pin="1234")),
        use(call("change_tariff", tariff_id="max")),
    ]
    paused = script.post(
        "/chats/", headers=HEADERS, json={"query": "Перейти на тариф Макс, номер 89990001122, PIN 1234"}
    ).json()
    assert paused["approval"]["actions"][0]["name"] == "change_tariff"
    assert "Макс" in paused["approval"]["actions"][0]["description"]
    assert paused["state"]["verified"] and crm.customer("+79990001122")["tariff"] == "smart"
    chat_id = paused["chat_id"]
    assert script.get(f"/chats/{chat_id}", headers=HEADERS).json()["waiting"]["answer_id"] == paused["answer_id"]

    script.model.script += [say("Готово: тариф «Макс».")]
    stale = script.post(
        f"/chats/{chat_id}/decision", headers=HEADERS, json={"decision": "approve", "approval_id": "old"}
    )
    assert stale.status_code == 409 and crm.customer("+79990001122")["tariff"] == "smart"
    done = script.post(f"/chats/{chat_id}/decision", headers=HEADERS, json={"decision": "approve"}).json()
    assert done["answer"].startswith("Готово") and done["approval"] is None
    assert crm.customer("+79990001122")["tariff"] == "max"
    # An action is done once: the same decision again has nothing to decide.
    assert script.post(f"/chats/{chat_id}/decision", headers=HEADERS, json={"decision": "approve"}).status_code == 409


def test_feedback_rename_and_delete(script):
    script.model.script += [say("Здравствуйте!")]
    answer = script.post("/chats/", headers=HEADERS, json={"query": "Привет"}).json()
    chat_id, answer_id = answer["chat_id"], answer["answer_id"]
    url = f"/chats/{chat_id}/answers/{answer_id}/feedback"
    assert script.post(url, headers=HEADERS, json={"feedback": "dislike", "comment": "долго"}).status_code == 204
    assert script.post(url, headers=OTHER, json={"feedback": "like"}).status_code == 404
    shown = script.get(f"/chats/{chat_id}", headers=HEADERS).json()["answers"][0]
    assert (shown["feedback"], shown["feedback_comment"]) == ("dislike", "долго")

    assert script.patch(f"/chats/{chat_id}", headers=HEADERS, json={"title": "Моё имя"}).json()["title"] == "Моё имя"
    assert script.delete(f"/chats/{chat_id}", headers=HEADERS).status_code == 204
    assert script.get(f"/chats/{chat_id}", headers=HEADERS).status_code == 404
