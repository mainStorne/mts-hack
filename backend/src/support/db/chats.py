"""The conversations of the API in Postgres, tables chats and answers (db/schema.sql): whose each chat is and its name, its
answers as the client got them with the customer's 👍/👎, and the checkpoint the agent's thread goes on from. A chat is named
by its first question until the model names it (agent/titles.py) or the customer renames it; the model never replaces the
customer's name. The agent's thread itself is in LangGraph's tables (AsyncPostgresSaver, api/runner.py `memory`).

The functions block: the API calls them with asyncio.to_thread.
"""

import uuid

from psycopg.types.json import Jsonb

from support.api.schemas import Answer, Chat, ChatHistory, SessionState
from support.db import pg

TITLE_MAX = 200  # characters of the first question kept as the chat's title
LISTED = 100  # chats in GET /chats


def new_id() -> str:
    return uuid.uuid4().hex[:24]


def thread_of(chat_id: str) -> str:
    """The agent's thread of the chat."""
    return f"api:{chat_id}"


def start(client_id: str, question: str) -> str:
    chat_id = new_id()
    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO chats (chat_id, client_id, title) VALUES (%s, %s, %s)",
            (chat_id, client_id, question.strip()[:TITLE_MAX]),
        )
    return chat_id


def owner(chat_id: str) -> str | None:
    with pg.connection() as conn:
        return pg.scalar(conn, "SELECT client_id FROM chats WHERE chat_id = %s", (chat_id,))


def checkpoint(chat_id: str) -> str | None:
    """The checkpoint of the agent's thread after the chat's last answer; None before the first one."""
    with pg.connection() as conn:
        return pg.scalar(conn, "SELECT checkpoint FROM chats WHERE chat_id = %s", (chat_id,))


def needs_title(chat_id: str) -> bool:
    with pg.connection() as conn:
        return bool(pg.scalar(conn, "SELECT title_by = 'question' FROM chats WHERE chat_id = %s", (chat_id,)))


def save_answer(response: Answer, span_id: str | None, checkpoint: str | None, title: str | None = None) -> None:
    """Keep an answer in its chat with the root span of its trace (what a vote annotates) and the checkpoint of the agent's
    thread after it, which the next question goes on from. `title` is the model's name of the chat: taken only while the chat
    is named by its question."""
    kept = response.model_dump(mode="json", exclude={"feedback", "feedback_comment"})
    with pg.connection() as conn:
        conn.execute(
            "INSERT INTO answers (chat_id, answer_id, span_id, response) VALUES (%s, %s, %s, %s)",
            (response.chat_id, response.answer_id, span_id, Jsonb(kept)),
        )
        conn.execute(
            "UPDATE chats SET checkpoint = %s, waiting_checkpoint = NULL, waiting = NULL, updated_at = now() WHERE chat_id = %s",
            (checkpoint, response.chat_id),
        )
        if title:
            conn.execute(
                "UPDATE chats SET title = %s, title_by = 'model' WHERE chat_id = %s AND title_by = 'question'",
                (title[:TITLE_MAX], response.chat_id),
            )


def save_waiting(response: Answer, checkpoint: str, interrupts: list[dict]) -> None:
    """Keep a question whose answer waits for the customer's decision: the checkpoint the agent paused at and its interrupts
    ({id, actions}: how many decisions each wants). The chat's own checkpoint stays that of its last answer: a new question
    goes on from it, as if this one had not been asked."""
    with pg.connection() as conn:
        conn.execute(
            "UPDATE chats SET waiting_checkpoint = %s, waiting = %s, updated_at = now() WHERE chat_id = %s",
            (
                checkpoint,
                Jsonb({"response": response.model_dump(mode="json"), "interrupts": interrupts}),
                response.chat_id,
            ),
        )


def waiting(chat_id: str) -> tuple[str, list[dict], Answer] | None:
    """(the checkpoint to resume, the interrupts, what the client got) of the chat's question waiting for a decision; None."""
    with pg.connection() as conn:
        row = conn.execute("SELECT waiting_checkpoint, waiting FROM chats WHERE chat_id = %s", (chat_id,)).fetchone()
    if row is None or not row["waiting_checkpoint"] or not row["waiting"]:
        return None
    return row["waiting_checkpoint"], row["waiting"]["interrupts"], Answer(**row["waiting"]["response"])


def take_waiting(chat_id: str, checkpoint: str) -> bool:
    """Take the question waiting at `checkpoint` for the decision now made: it waits no more, whether the agent then answers or
    fails, so that an action is never done twice. False when it is not waiting there any more (decided meanwhile)."""
    with pg.connection() as conn:
        return (
            conn.execute(
                "UPDATE chats SET waiting_checkpoint = NULL, waiting = NULL WHERE chat_id = %s AND waiting_checkpoint = %s",
                (chat_id, checkpoint),
            ).rowcount
            > 0
        )


def rename(chat_id: str, client_id: str, title: str) -> Chat | None:
    """The customer's name of their chat, which the model does not replace; None when the chat is not theirs."""
    with pg.connection() as conn:
        row = conn.execute(
            """UPDATE chats SET title = %s, title_by = 'user' WHERE chat_id = %s AND client_id = %s
               RETURNING chat_id, title, created_at, updated_at""",
            (title.strip()[:TITLE_MAX], chat_id, client_id),
        ).fetchone()
    return None if row is None else Chat(**row)


def delete(chat_id: str, client_id: str) -> bool:
    """The chat goes with its answers (ON DELETE CASCADE); False when it is not theirs. The agent's thread is the caller's."""
    with pg.connection() as conn:
        return (
            conn.execute("DELETE FROM chats WHERE chat_id = %s AND client_id = %s", (chat_id, client_id)).rowcount > 0
        )


def of_client(client_id: str) -> list[Chat]:
    """The customer's chats with an answer or a decision waiting, most recently updated first."""
    with pg.connection() as conn:
        rows = conn.execute(
            """SELECT chat_id, title, created_at, updated_at FROM chats c
               WHERE client_id = %s AND (c.waiting IS NOT NULL OR EXISTS (SELECT 1 FROM answers a WHERE a.chat_id = c.chat_id))
               ORDER BY updated_at DESC LIMIT %s""",
            (client_id, LISTED),
        ).fetchall()
    return [Chat(**row) for row in rows]


def history(chat_id: str, client_id: str) -> ChatHistory | None:
    """The chat with its answers, the oldest first; None when it is not the customer's."""
    with pg.connection() as conn:
        chat = conn.execute(
            "SELECT chat_id, title, created_at, updated_at, waiting FROM chats WHERE chat_id = %s AND client_id = %s",
            (chat_id, client_id),
        ).fetchone()
        if chat is None:
            return None
        rows = conn.execute(
            "SELECT response, feedback, comment FROM answers WHERE chat_id = %s ORDER BY id", (chat_id,)
        ).fetchall()
    answers = [Answer(**row["response"], feedback=row["feedback"], feedback_comment=row["comment"]) for row in rows]
    pending = chat.pop("waiting")
    waiting_for = Answer(**pending["response"]) if pending else None
    last = waiting_for or (answers[-1] if answers else None)
    return ChatHistory(**chat, state=last.state if last else SessionState(), answers=answers, waiting=waiting_for)


def asked(chat_id: str, answer_id: str) -> tuple[str, str | None] | None:
    """(the owner of the chat, the root span of the answer's trace) of an answer; None when there is none."""
    with pg.connection() as conn:
        row = conn.execute(
            "SELECT c.client_id, a.span_id FROM answers a JOIN chats c USING (chat_id) WHERE a.chat_id = %s AND a.answer_id = %s",
            (chat_id, answer_id),
        ).fetchone()
    return None if row is None else (row["client_id"], row["span_id"])


def vote(answer_id: str, feedback: str | None, comment: str | None) -> None:
    """The customer's 👍/👎 on an answer with what they said of it; it replaces the one before."""
    with pg.connection() as conn:
        conn.execute(
            "UPDATE answers SET feedback = %s, comment = %s WHERE answer_id = %s", (feedback, comment, answer_id)
        )
