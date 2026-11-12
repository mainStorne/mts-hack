"""What the conversation remembers between messages, checkpointed with the thread (Postgres in the API)."""

from dataclasses import dataclass
from typing import NotRequired

from langchain.agents import AgentState


class SupportState(AgentState):
    active_agent: NotRequired[str]  # who speaks; agent/agents.py START until the first handoff
    verified: NotRequired[bool]  # verify_customer succeeded: account tools are available
    customer_phone: NotRequired[str | None]
    customer_name: NotRequired[str | None]
    ticket_id: NotRequired[str | None]  # set when the conversation is handed to a human
    escalated: NotRequired[bool]
    verify_failures: NotRequired[int]  # wrong PINs in a row; too many and verification is refused


@dataclass
class AgentContext:
    """Per-request data that is not conversation state."""

    client_id: str | None = None
