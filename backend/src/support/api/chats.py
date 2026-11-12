"""The routes of the assistant: ask (JSON or server-sent events), decide, cancel, the customer's chats, feedback."""

import asyncio
import json
import logging
import time

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from support.agent import tracing
from support.api import runner
from support.api.auth import ClientDep
from support.api.schemas import Answer, AskBody, Chat, ChatHistory, Decision, FeedbackBody, RenameBody
from support.common.llm_down import UNAVAILABLE, unreachable
from support.db import chats

log = logging.getLogger("support.api")
router = APIRouter(prefix="/chats", tags=["Chats"])

NO_CHAT = "Chat not found or the client has no access to it"
NOTHING_WAITS = "Nothing in this chat waits for a decision (it was decided already, or a newer question was answered)"
BUSY = "The chat is answering another question"
ERRORS = {
    404: {"description": NO_CHAT},
    409: {"description": "The chat is busy, nothing waits for a decision, or the question was cancelled"},
    502: {"description": "ML service error"},
    503: {"description": "The language model cannot be reached; the detail says the assistant is not working now"},
}


async def chat_of(client_id: str, chat_id: str | None, query: str) -> str:
    """The chat to ask in: a new one without `chat_id`; 404 for a chat that is not the client's."""
    if chat_id is None:
        return await asyncio.to_thread(chats.start, client_id, query)
    if await asyncio.to_thread(chats.owner, chat_id) != client_id:
        raise HTTPException(404, NO_CHAT)
    return chat_id


def server_event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _failure(error: Exception) -> HTTPException:
    if isinstance(error, runner.Busy):
        return HTTPException(409, BUSY)
    if isinstance(error, runner.NothingToDecide):
        return HTTPException(409, NOTHING_WAITS)
    if isinstance(error, runner.AskCancelled):
        return HTTPException(409, "The question was cancelled")
    log.warning("the agent failed (%s)", type(error).__name__)
    return HTTPException(503, UNAVAILABLE) if unreachable(error) else HTTPException(502, "ML service error")


def _streamed(query: str, chat_id: str, client_id: str, decision: Decision | None = None) -> StreamingResponse:
    """The answer to `query` (with `decision`: to the question that waited for it) as server-sent events."""
    events: asyncio.Queue = asyncio.Queue()

    async def work() -> None:
        started = time.monotonic()
        try:
            done = await runner.run(query, chat_id, client_id, events, decision)
            log.info("ask stream completed (%d ms)", round((time.monotonic() - started) * 1000))
            await events.put(("done", done.model_dump(mode="json")))
        except Exception as e:  # noqa: BLE001 - every failure is an event: the response is already 200
            failure = _failure(e)
            await events.put(
                ("cancelled" if isinstance(e, runner.AskCancelled) else "error", {"detail": failure.detail})
            )
        await events.put(None)

    async def generate():
        yield server_event("chat", {"chat_id": chat_id})
        task = asyncio.create_task(work())
        try:
            while (event := await events.get()) is not None:
                yield server_event(*event)
        finally:
            task.cancel()  # the client went away: the answer is not needed any more

    return StreamingResponse(
        generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@router.post("", response_model=Answer, responses=ERRORS, include_in_schema=False)
@router.post("/", response_model=Answer, responses=ERRORS)
async def ask(client_id: ClientDep, body: AskBody) -> Answer:
    """Ask the assistant and wait for the whole answer."""
    chat_id = await chat_of(client_id, body.chat_id, body.query)
    try:
        return await runner.run(body.query, chat_id, client_id)
    except Exception as e:
        raise _failure(e) from e


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent events: `chat` {chat_id}, `status` {text}, `agent` {name, label} (a handoff), "
            "`answer` {text} (the whole answer so far), then `done` (an Answer), `error` or `cancelled` {detail}"
        }
    },
)
async def ask_stream(client_id: ClientDep, body: AskBody):
    """Ask the assistant and watch it work: statuses, handoffs between specialists and the answer as it is written."""
    chat_id = await chat_of(client_id, body.chat_id, body.query)
    return _streamed(body.query, chat_id, client_id)


async def _waiting(client_id: str, chat_id: str) -> None:
    if await asyncio.to_thread(chats.owner, chat_id) != client_id:
        raise HTTPException(404, NO_CHAT)
    if await asyncio.to_thread(chats.waiting, chat_id) is None:
        raise HTTPException(409, NOTHING_WAITS)


@router.post("/{chat_id}/decision", response_model=Answer, responses=ERRORS)
async def decide(client_id: ClientDep, chat_id: str, decision: Decision) -> Answer:
    """Approve or reject what the assistant is about to do (the `approval` of the chat's waiting answer). It goes on from where
    it paused, does the action only after an approval, and answers the question that waited."""
    await _waiting(client_id, chat_id)
    try:
        return await runner.run("", chat_id, client_id, decision=decision)
    except Exception as e:
        raise _failure(e) from e


@router.post("/{chat_id}/decision/stream", response_class=StreamingResponse)
async def decide_stream(client_id: ClientDep, chat_id: str, decision: Decision):
    """The same as POST /chats/{chat_id}/decision, with the events of POST /chats/stream."""
    await _waiting(client_id, chat_id)
    return _streamed("", chat_id, client_id, decision)


@router.post("/{chat_id}/cancel", status_code=204, responses={404: {"description": NO_CHAT}})
async def cancel(client_id: ClientDep, chat_id: str) -> Response:
    """Cancel the question being answered in the chat. Its request ends with 409 (`/stream`: a `cancelled` event); the
    conversation goes on as it was before it."""
    if await asyncio.to_thread(chats.owner, chat_id) != client_id:
        raise HTTPException(404, NO_CHAT)
    runner.cancel(chat_id)
    return Response(status_code=204)


@router.get("", response_model=list[Chat], include_in_schema=False)
@router.get("/", response_model=list[Chat])
async def list_chats(client_id: ClientDep) -> list[Chat]:
    """The client's chats, most recently updated first."""
    return await asyncio.to_thread(chats.of_client, client_id)


@router.get("/{chat_id}", response_model=ChatHistory, responses={404: {"description": NO_CHAT}})
async def chat_history(client_id: ClientDep, chat_id: str) -> ChatHistory:
    """The chat with its answers (and the question that waits for a decision), to come back to it."""
    found = await asyncio.to_thread(chats.history, chat_id, client_id)
    if found is None:
        raise HTTPException(404, NO_CHAT)
    return found


@router.patch("/{chat_id}", response_model=Chat, responses={404: {"description": NO_CHAT}})
async def rename_chat(client_id: ClientDep, chat_id: str, body: RenameBody) -> Chat:
    found = await asyncio.to_thread(chats.rename, chat_id, client_id, body.title)
    if found is None:
        raise HTTPException(404, NO_CHAT)
    return found


@router.delete("/{chat_id}", status_code=204, responses={404: {"description": NO_CHAT}})
async def delete_chat(client_id: ClientDep, chat_id: str) -> Response:
    """Delete the chat with the agent's memory of it."""
    runner.cancel(chat_id)
    if not await asyncio.to_thread(chats.delete, chat_id, client_id):
        raise HTTPException(404, NO_CHAT)
    await runner.agent().checkpointer.adelete_thread(chats.thread_of(chat_id))
    return Response(status_code=204)


@router.post(
    "/{chat_id}/answers/{answer_id}/feedback", status_code=204, responses={404: {"description": "Answer not found"}}
)
async def add_feedback(client_id: ClientDep, chat_id: str, answer_id: str, body: FeedbackBody) -> Response:
    """👍/👎 on an answer, with what the customer liked or not; it is also an annotation on the answer's trace in Phoenix."""
    found = await asyncio.to_thread(chats.asked, chat_id, answer_id)
    if found is None or found[0] != client_id:
        raise HTTPException(404, "Answer not found or the client has no access to it")
    await asyncio.to_thread(chats.vote, answer_id, body.feedback, body.comment)
    if body.feedback and found[1] and tracing.enabled():
        try:
            await asyncio.to_thread(
                tracing.score_feedback, found[1], f"api:{client_id}", body.feedback == "like", body.comment
            )
        except Exception as e:  # noqa: BLE001 - the vote is kept; Phoenix is a convenience
            log.warning("could not annotate the trace (%s)", type(e).__name__)
    return Response(status_code=204)
