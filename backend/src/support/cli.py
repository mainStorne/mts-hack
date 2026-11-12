"""`support` — talk to the agent in the terminal (memory of the process, no database): handy for trying a prompt or a tool.

uv run support              # a chat; the agent speaks to the model of .env
uv run support "вопрос"     # one question
"""

import asyncio
import logging
import sys
import uuid

from support.agent.core import build_agent, final_answer, session_state, turn_handoffs
from support.agent.state import AgentContext


async def ask(agent, thread: str, text: str) -> None:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": text}]},
        {"configurable": {"thread_id": thread}},
        context=AgentContext(),
    )
    for h in turn_handoffs(result["messages"]):
        print(f"  ↪ {h['from_agent']} → {h['to_agent']}: {h['reason']}")
    state = session_state(result)
    print(
        f"[{state['active_agent']}{' · verified' if state['verified'] else ''}{' · ticket ' + state['ticket_id'] if state['ticket_id'] else ''}]"
    )
    print(final_answer(result["messages"]))
    snapshot = await agent.aget_state({"configurable": {"thread_id": thread}})
    for task in snapshot.tasks:
        for interrupt in task.interrupts:
            for action in interrupt.value.get("action_requests", []):
                print(f"  ⏸ waits for your approval: {action['description']} (the API resumes it; the CLI cannot)")


async def chat(first: str | None) -> None:
    agent, thread = build_agent(), uuid.uuid4().hex
    text = first
    while True:
        if text is None:
            try:
                text = input("> ").strip()
            except EOFError:
                return
        if text:
            await ask(agent, thread, text)
        if first is not None:
            return
        text = None


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(chat(" ".join(sys.argv[1:]) or None))
