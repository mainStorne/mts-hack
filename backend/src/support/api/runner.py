"""Runs the agent for the API: the agent with its memory in Postgres, one question at a time per chat, streaming, the pause
for the customer's decision, cancelling."""

import asyncio
import contextlib
import logging

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from support.agent import titles, tracing
from support.agent.agents import AGENTS, START
from support.agent.core import DECLINED, SUPERVISOR_NAME, build_agent, final_answer, session_state, turn_handoffs
from support.agent.state import AgentContext
from support.api.schemas import Answer, Approval, ApprovalAction, Decision, Handoff, SessionState
from support.common.lang import language
from support.common.render import partial_text
from support.common.status import CHECKING, STATUS, WRITING
from support.db import chats
from support.settings import settings

log = logging.getLogger("support.api")

_agent = None
TITLE_WAIT = 5  # seconds an answer waits for the model's name of the chat
WAITING = {"Russian": "Подтвердите действие, и я продолжу.", "English": "Please confirm the action and I will go on."}


def agent():
    """The agent of the API, built when the app starts (`memory`)."""
    if _agent is None:
        raise RuntimeError("the agent is built when the app starts (api/runner.py `memory`)")
    return _agent


@contextlib.asynccontextmanager
async def memory():
    """For the life of the app: the agent that keeps its memory of the chats in Postgres (AsyncPostgresSaver, thread
    `api:<chat_id>`; `setup` makes its tables or brings them up to date)."""
    global _agent
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set (see .env.example)")
    pool = AsyncConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=2,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    async with pool:
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]  # (a pool of dict-row connections, as it wants)
        await saver.setup()
        _agent = build_agent(checkpointer=saver)
        try:
            yield
        finally:
            _agent = None


class AskCancelled(Exception):
    """The question was cancelled (POST /chats/{id}/cancel)."""


class NothingToDecide(Exception):
    """No question of the chat waits for this decision."""


class Busy(Exception):
    """The chat is answering another question."""


pending: dict[str, asyncio.Task] = {}  # chat_id -> the question being answered


def _approval(interrupts, checkpoint: str) -> Approval:
    actions = []
    for interrupt in interrupts:
        value = interrupt.value if isinstance(interrupt.value, dict) else {}
        for request in value.get("action_requests") or [{"name": "action", "description": str(interrupt.value)}]:
            actions.append(
                ApprovalAction(name=request.get("name") or "action", description=request.get("description") or "")
            )
    return Approval(id=checkpoint, actions=actions)


def _decisions(found: list[dict], decision: str) -> dict:
    """The resume of each interrupt: one decision per action it asked about, the same for all."""
    one = {"type": "approve"} if decision == "approve" else {"type": "reject", "message": DECLINED}
    return {i["id"]: {"decisions": [one] * i["actions"]} for i in found}


async def stream_agent(inputs, config: dict, context: AgentContext, events: asyncio.Queue) -> dict:
    """Run the agent, queueing statuses, handoffs and the whole answer written so far (each `answer` event replaces the last)."""
    state: dict = {"messages": []}
    active = None
    call_id, call_text, shown, status = None, "", "", CHECKING

    async def show_status(text: str | None) -> None:
        nonlocal status
        if text and text != status:
            status = text
            await events.put(("status", {"text": text}))

    await events.put(("status", {"text": CHECKING}))
    async for mode, data in agent().astream(inputs, config, context=context, stream_mode=["messages", "values"]):
        if mode == "values":
            if (
                "messages" not in data
            ):  # (not the interrupt the run paused on: api/chats.py reads it from the thread's state)
                continue
            state = data
            now = data.get("active_agent") or START
            if active is not None and now != active:
                await events.put(("agent", {"name": now, "label": AGENTS[now].label}))
                await show_status(WRITING)
            active = now
            continue
        chunk, metadata = data
        if isinstance(chunk, ToolMessage):
            continue
        if not isinstance(chunk, AIMessageChunk):
            continue
        if metadata.get("langgraph_node") != "model" or metadata.get("lc_agent_name") != SUPERVISOR_NAME:
            continue
        if chunk.tool_call_chunks:
            await show_status(STATUS.get(chunk.tool_call_chunks[0].get("name") or ""))
            continue
        if chunk.id != call_id:  # a new model call begins: its text starts from zero
            call_id, call_text = chunk.id, ""
        call_text += chunk.text
        text = partial_text(call_text)
        if text and text != shown:
            shown = text
            await events.put(("answer", {"text": text}))
    return state


async def run(
    query: str, chat_id: str, client_id: str, events: asyncio.Queue | None = None, decision: Decision | None = None
) -> Answer:
    """Answer a question and keep the answer in its chat; with `decision`, go on with the question that waits for it (`query`
    is ignored). It runs as a task of its own that `cancel` can cancel: then `AskCancelled` is raised here, while the
    cancelling of the caller itself (the client went away) goes on as it is."""
    if (running := pending.get(chat_id)) and not running.done():
        raise Busy(chat_id)
    task = asyncio.ensure_future(_answer(query, chat_id, client_id, events, decision))
    pending[chat_id] = task
    task.add_done_callback(lambda done: pending.get(chat_id) is done and pending.pop(chat_id))
    try:
        return await task
    except asyncio.CancelledError:
        if task.cancelled() and not asyncio.current_task().cancelling():
            raise AskCancelled(chat_id) from None
        raise


def cancel(chat_id: str) -> bool:
    task = pending.get(chat_id)
    if task is None or task.done():
        return False
    task.cancel()
    return True


async def chat_title(question: str) -> str | None:
    """The model's name of a chat that starts with the question. A bonus: when it fails, the chat keeps the question."""
    try:
        return await asyncio.to_thread(titles.name, question)
    except Exception as exc:  # noqa: BLE001 - the answer does not need it
        log.warning("could not name the chat (%s)", type(exc).__name__)
        return None


async def _answer(
    query: str, chat_id: str, client_id: str, events: asyncio.Queue | None, decision: Decision | None
) -> Answer:
    thread_id = chats.thread_of(chat_id)
    config = tracing.run_config(thread_id, client_id)
    last = await asyncio.to_thread(chats.checkpoint, chat_id)
    resume = None
    if decision is not None:
        # The decision goes on from the checkpoint the agent paused at, and only from the one the customer saw.
        found = await asyncio.to_thread(chats.waiting, chat_id)
        if found is None or (decision.approval_id is not None and decision.approval_id != found[0]):
            raise NothingToDecide(chat_id)
        config["configurable"]["checkpoint_id"] = found[0]
        resume = Command(resume=_decisions(found[1], decision.decision))
        query = found[2].query
    # A question goes on from the chat's last answer: one that was cancelled (or failed) halfway left its messages in the
    # thread, and a tool call without its result would break the next question. Before the first answer there is nothing to
    # go on from: what such a question left goes.
    elif last:
        config["configurable"]["checkpoint_id"] = last
    else:
        await agent().checkpointer.adelete_thread(thread_id)

    # A chat not named yet is named while the agent answers: the answer does not wait for it.
    naming = (
        asyncio.ensure_future(chat_title(query))
        if decision is None and await asyncio.to_thread(chats.needs_title, chat_id)
        else None
    )
    try:
        with tracing.question(thread_id, client_id, query) as traced:
            inputs = resume or {"messages": [{"role": "user", "content": query}]}
            # Taken right before the agent goes on: from here the decision cannot be made again (chats.take_waiting).
            if resume is not None and not await asyncio.to_thread(
                chats.take_waiting, chat_id, config["configurable"]["checkpoint_id"]
            ):
                raise NothingToDecide(chat_id)
            context = AgentContext(client_id=client_id)
            if events is None:
                result = await agent().ainvoke(inputs, config, context=context)
            else:
                result = await stream_agent(inputs, config, context, events)
            after = await agent().aget_state(
                {"configurable": {"thread_id": thread_id}}
            )  # the thread's latest checkpoint
            state = SessionState(**session_state(after.values or result))
            handoffs = [Handoff(**h) for h in turn_handoffs(result["messages"])]
            if paused := list(getattr(after, "interrupts", None) or ()):
                checkpoint = after.config["configurable"]["checkpoint_id"]
                text = (
                    final_answer(result["messages"])
                    if _spoke_before_pausing(result["messages"])
                    else WAITING[language(query)]
                )
                traced.answer(text)
                waiting = Answer(
                    answer_id=chats.new_id(), chat_id=chat_id, query=query, answer=text, state=state, handoffs=handoffs,
                    approval=_approval(paused, checkpoint),
                )  # fmt: skip
                counts = [
                    {"id": i.id, "actions": len((i.value or {}).get("action_requests") or [None])} for i in paused
                ]
                if naming is not None:
                    naming.cancel()  # named when it is answered
                await asyncio.to_thread(chats.save_waiting, waiting, checkpoint, counts)
                return waiting
            text = final_answer(result["messages"])
            traced.answer(text)
    except BaseException:
        if naming is not None:
            naming.cancel()  # the question is not kept: the next one names the chat
        raise
    response = Answer(
        answer_id=chats.new_id(), chat_id=chat_id, query=query, answer=text, state=state, handoffs=handoffs
    )
    title = None
    if naming is not None:
        done, _ = await asyncio.wait({naming}, timeout=TITLE_WAIT)
        title = naming.result() if done else None
        naming.cancel()
    await asyncio.to_thread(
        chats.save_answer, response, traced.id, after.config["configurable"]["checkpoint_id"], title
    )
    return response


def _spoke_before_pausing(messages: list) -> bool:
    """A tool call can carry the explanation of what the customer is about to approve: the last model message of the turn."""
    return bool(messages) and isinstance(messages[-1], AIMessage) and bool(messages[-1].text.strip())
