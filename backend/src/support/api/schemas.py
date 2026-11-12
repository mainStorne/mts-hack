"""The JSON of the API (also the contract of the frontend)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

AgentName = Literal["triage", "billing", "tariffs", "tech", "human"]


class SessionState(BaseModel):
    """What the conversation remembers: who speaks and what is known about the customer."""

    active_agent: AgentName = "triage"
    verified: bool = False
    customer_name: str | None = None
    ticket_id: str | None = None
    "Set when the conversation was handed to a human operator."
    escalated: bool = False


class Handoff(BaseModel):
    from_agent: AgentName
    to_agent: AgentName
    reason: str


class ApprovalAction(BaseModel):
    name: str
    description: str


class Approval(BaseModel):
    id: str
    "The checkpoint the agent paused at; POST /chats/{id}/decision answers it."
    actions: list[ApprovalAction]


class Answer(BaseModel):
    answer_id: str
    chat_id: str
    query: str
    answer: str
    state: SessionState = SessionState()
    handoffs: list[Handoff] = []
    "The handoffs made while answering this question, in order."
    approval: Approval | None = None
    "The agent waits for the customer to approve or reject these actions."
    feedback: Literal["like", "dislike"] | None = None
    feedback_comment: str | None = None


class Chat(BaseModel):
    chat_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ChatHistory(Chat):
    state: SessionState = SessionState()
    answers: list[Answer]
    waiting: Answer | None = None


NOT_BLANK = r"\S"


class AskBody(BaseModel):
    query: str = Field(..., pattern=NOT_BLANK, max_length=4000)
    chat_id: str | None = None
    "Omit to start a new chat."


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    approval_id: str | None = None
    "The `approval.id` the customer saw; a decision on another one is refused."


class FeedbackBody(BaseModel):
    feedback: Literal["like", "dislike"] | None
    comment: str | None = Field(None, max_length=2000)


class RenameBody(BaseModel):
    title: str = Field(..., pattern=NOT_BLANK, max_length=200)
