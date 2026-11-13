"""The support agent: one LangChain/LangGraph agent whose behaviour is switched by `active_agent` in its state (handoffs).

    customer message ─► guardrail ─► model call as the ACTIVE specialist (its prompt, its tools)
                                       │ tool call transfer_to_<x>  ─► `active_agent` = x, the loop goes on as x
                                       │ tool call change_tariff    ─► pauses for the customer's approval (HITL)
                                       └ plain text                 ─► the answer; the state is checkpointed

The next message of the chat starts from the saved state, so it goes to the specialist that was active.
"""

import logging
import time
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    SummarizationMiddleware,
    ToolCallRequest,
    ToolErrorMiddleware,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from support import crm
from support.agent.agents import ACCOUNT_TOOLS, AGENTS, COMMON, START
from support.agent.guardrail import GuardrailMiddleware
from support.agent.state import AgentContext, SupportState
from support.agent.tools import ALL_TOOLS
from support.common.lang import language
from support.settings import settings

log = logging.getLogger("support.agent")

SUPERVISOR_NAME = "support"
DECLINED = "The customer declined: nothing was changed. Do not ask for it again unless they ask."
NO_ANSWER = "Не получилось сформулировать ответ. Попробуйте переформулировать вопрос."
SUMMARY_PROMPT = """\
Summarize this conversation between a customer and a mobile operator's support so that the next specialist can continue it. \
Keep: what the customer wants, their verified name if any, facts found (tariffs, charges, outages), what was done, what is \
still open. Drop greetings. At most 150 words, in the language of the conversation. Reply with the summary only.

Conversation:
{messages}
"""


def _model(**overrides) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=str(settings.llm_route),
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        **{
            "max_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": settings.llm_thinking}},
            **overrides,
        },
    )


def session_facts(state: dict) -> str:
    """What the specialist knows about the session, appended to its prompt."""
    lines = ["Session facts:"]
    if state.get("verified"):
        phone = str(state.get("customer_phone") or "")
        lines.append(
            f"- Customer verified: {state.get('customer_name')} (number ending {phone[-4:]}). Account tools are available."
        )
    else:
        lines.append("- Customer is NOT verified: account tools are unavailable until verify_customer succeeds.")
    if state.get("ticket_id"):
        lines.append(f"- Human operator ticket: {state['ticket_id']}.")
    return "\n".join(lines)


class HandoffMiddleware(AgentMiddleware[SupportState, AgentContext]):
    """Gives every model call the prompt and the tools of the active specialist.

    Account tools are offered only to a verified customer. Parallel tool calls are off: two handoffs at once would conflict."""

    state_schema = SupportState

    @staticmethod
    def specialist(state: dict):
        return AGENTS.get(state.get("active_agent") or START, AGENTS[START])

    def _for_active(self, request: ModelRequest) -> ModelRequest:
        state: dict[str, Any] = request.state
        agent = self.specialist(state)
        names = set(agent.tools) | {f"transfer_to_{target}" for target in agent.handoffs}
        if state.get("verified"):
            names.discard("verify_customer")
        else:
            names -= ACCOUNT_TOOLS
        tools = [t for t in request.tools if getattr(t, "name", None) in names]
        # The stable part of the prompt first: the model server keeps its cache for it.
        prompt = "\n\n".join(
            [COMMON, f"You are now the {agent.label} specialist.\n{agent.prompt}", session_facts(state)]
        )
        return request.override(
            system_message=SystemMessage(content=prompt),
            tools=tools,
            model_settings={**request.model_settings, "parallel_tool_calls": False},
        )

    def wrap_model_call(self, request, handler):
        return handler(self._for_active(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._for_active(request))


class LoggingMiddleware(AgentMiddleware):
    """Logs the active specialist, context size and duration of every model call, and every tool call."""

    def _log_model(self, request: ModelRequest, started: float) -> None:
        log.info(
            "model call as %s: %d messages, ~%d tokens in context, %.1fs",
            request.state.get("active_agent") or START,
            len(request.messages),
            count_tokens_approximately(request.messages),
            time.monotonic() - started,
        )

    @staticmethod
    def _log_tool(request: ToolCallRequest, started: float) -> None:
        call = request.tool_call
        log.info("tool %s(%s): %.1fs", call["name"], call["args"], time.monotonic() - started)

    async def awrap_model_call(self, request, handler):
        started = time.monotonic()
        response = await handler(request)
        self._log_model(request, started)
        return response

    def wrap_model_call(self, request, handler):
        started = time.monotonic()
        response = handler(request)
        self._log_model(request, started)
        return response

    async def awrap_tool_call(self, request, handler):
        started = time.monotonic()
        result = await handler(request)
        self._log_tool(request, started)
        return result

    def wrap_tool_call(self, request, handler):
        started = time.monotonic()
        result = handler(request)
        self._log_tool(request, started)
        return result


def _tool_error(error: Exception, request: ToolCallRequest) -> str:
    """Hand the failure back to the model so it can fix the arguments or answer without the tool."""
    return f"Tool error: {type(error).__name__}: {error}"


def _describe_tariff_change(tool_call, state, runtime) -> str:
    tariff = crm.tariff(str(tool_call["args"].get("tariff_id")))
    question = next((str(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    if tariff is None:
        return "Сменить тариф"
    if language(question) == "Russian":
        return f"Сменить тариф на «{tariff['name']}» ({tariff['price']} ₽ в месяц). Остатки пакетов минут и ГБ сгорят."
    return f"Switch to the {tariff['name']} tariff ({tariff['price']} RUB a month). Unused minutes and data are lost."


# Actions that change the account wait for the customer's decision (api/chats.py reads the interrupt and resumes the thread).
APPROVAL = {"change_tariff": {"allowed_decisions": ["approve", "reject"], "description": _describe_tariff_change}}


def build_agent(model=None, *, guardrail: bool | None = None, checkpointer: BaseCheckpointSaver | None = None):
    """The agent. `guardrail=False` leaves out the screen in front of it; tests do that. The conversations are remembered by
    `checkpointer`: the API's are in Postgres (api/chats.py `memory`), the CLI's and the tests' in the memory of the process."""
    model = model or _model()
    use_guardrail = settings.guardrail_enabled if guardrail is None else guardrail
    return create_agent(
        model,
        ALL_TOOLS,
        # Order matters: the first middleware wraps the rest.
        middleware=[
            *([GuardrailMiddleware()] if use_guardrail else []),  # first: a blocked message costs the agent nothing
            SummarizationMiddleware(
                _model(max_tokens=400, temperature=0, extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
                trigger=("tokens", settings.agent_summarize_trigger_tokens),
                keep=("messages", settings.agent_summarize_keep_messages),
                summary_prompt=SUMMARY_PROMPT,
            ),
            HandoffMiddleware(),
            HumanInTheLoopMiddleware(APPROVAL),
            LoggingMiddleware(),
            ToolErrorMiddleware(on_error=_tool_error),
            ModelCallLimitMiddleware(run_limit=settings.agent_max_model_calls, exit_behavior="end"),
        ],
        context_schema=AgentContext,
        state_schema=SupportState,
        checkpointer=checkpointer or InMemorySaver(),
        name=SUPERVISOR_NAME,  # metadata of its model calls says whose text it is (api/chats.py stream_agent)
    )


# --- reading the result ---------------------------------------------------------------------------------------------


def _this_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    last = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), -1)
    return messages[last + 1 :]


def final_answer(messages: list[BaseMessage]) -> str:
    """The text of the last model message of this turn that is not a tool call; the default when there is none."""
    for message in reversed(_this_turn(messages)):
        if isinstance(message, AIMessage) and not message.tool_calls and message.text.strip():
            return message.text.strip()
    return NO_ANSWER


def turn_handoffs(messages: list[BaseMessage]) -> list[dict]:
    """The handoffs made in this turn, in order: {from_agent, to_agent, reason}."""
    return [
        m.artifact
        for m in _this_turn(messages)
        if isinstance(m, ToolMessage) and isinstance(m.artifact, dict) and "to_agent" in m.artifact
    ]


def session_state(values: dict) -> dict:
    """The part of the graph state the clients see."""
    return {
        "active_agent": values.get("active_agent") or START,
        "verified": bool(values.get("verified")),
        "customer_name": values.get("customer_name"),
        "ticket_id": values.get("ticket_id"),
        "escalated": bool(values.get("escalated")),
    }


def history(messages: list[BaseMessage]) -> list[dict]:
    """The conversation as plain messages: the questions and the answers, without the tool work."""
    return [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.text}
        for m in messages
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls and m.text)
    ]
