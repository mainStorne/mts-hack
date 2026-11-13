"""The tools of all specialists. Which of them a specialist sees is decided in agent/core.py (HandoffMiddleware) from
agent/agents.py; the checks here (verification, allowed handoffs) hold even if the model calls a tool it was not offered.

Data comes from `support.crm`, a mock of the back office: replace it with your real systems.
"""

import json

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

from support import crm, retrieval
from support.agent.agents import ACCOUNT_TOOLS, AGENTS, START
from support.agent.state import AgentContext

MAX_VERIFY_FAILURES = 3
Runtime = ToolRuntime[AgentContext]


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _reply(runtime: Runtime, content: str, **update) -> Command:
    return Command(update={**update, "messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})


def _phone(runtime: Runtime) -> str | None:
    """The verified customer's phone from the state; None when nobody is verified."""
    state = runtime.state
    return state.get("customer_phone") if state.get("verified") else None


NOT_VERIFIED = (
    "Not allowed: the customer is not verified. Ask for the phone number and the PIN and call verify_customer."
)


# --- handoffs -------------------------------------------------------------------------------------------------------


def _handoff_tool(target: str) -> BaseTool:
    label = AGENTS[target].label

    @tool(
        f"transfer_to_{target}",
        description=(
            f"Hand the conversation over to the {label} specialist. `reason`: what the customer needs and what is already "
            "known, in one or two sentences, so that they do not have to repeat themselves."
        ),
    )
    def transfer(reason: str, runtime: Runtime) -> Command:
        source = runtime.state.get("active_agent") or START
        if target not in AGENTS[source].handoffs:
            return _reply(
                runtime, f"Not allowed: {AGENTS[source].label} cannot hand over to {label}. Continue yourself."
            )
        update: dict = {"active_agent": target}
        if target == "human":
            ticket_id = runtime.state.get("ticket_id")
            if not ticket_id:
                ticket_id = crm.create_ticket(runtime.state.get("customer_phone"), reason).id
            update |= {"ticket_id": ticket_id, "escalated": True}
            note = f"Handed over to a human operator, ticket {ticket_id}."
        else:
            note = f"Handed over to the {label} specialist. Continue the conversation as that specialist."
        message = ToolMessage(
            content=note,
            tool_call_id=runtime.tool_call_id,
            name=f"transfer_to_{target}",
            artifact={"from_agent": source, "to_agent": target, "reason": reason},
        )
        return Command(update={**update, "messages": [message]})

    return transfer


HANDOFF_TOOLS = [_handoff_tool(name) for name in AGENTS if name != START] + [_handoff_tool(START)]


# --- shared ---------------------------------------------------------------------------------------------------------


@tool
def verify_customer(phone: str, pin: str, runtime: Runtime) -> Command:
    """Verify the customer's identity by the phone number and the PIN they say. Required before any account data."""
    failures = runtime.state.get("verify_failures") or 0
    if failures >= MAX_VERIFY_FAILURES:
        return _reply(runtime, "Verification is blocked after too many wrong PINs. Do not retry: transfer_to_human.")
    found = crm.verify(phone, pin)
    if found is None:
        return _reply(
            runtime,
            "Verification failed: the phone number or the PIN is wrong. Do not say which one. Let the customer try again.",
            verify_failures=failures + 1,
        )
    return _reply(
        runtime,
        f"Verified: {found['name']}. Account tools are available now.",
        verified=True,
        customer_phone=found["phone"],
        customer_name=found["name"],
        verify_failures=0,
    )


@tool
def search_knowledge_base(query: str) -> str:
    """Search the support knowledge base (how-to articles and policies). `query`: a few keywords in Russian."""
    return _json(retrieval.search(query) or "Nothing found.")


# --- billing --------------------------------------------------------------------------------------------------------


@tool
def get_balance(runtime: Runtime) -> str:
    """The verified customer's balance in rubles (negative: they owe)."""
    if not (phone := _phone(runtime)):
        return NOT_VERIFIED
    return _json({"balance_rub": crm.customer(phone)["balance"]})


@tool
def get_recent_charges(runtime: Runtime) -> str:
    """The verified customer's recent charges: id, date, description and amount in rubles."""
    if not (phone := _phone(runtime)):
        return NOT_VERIFIED
    return _json(crm.customer(phone)["charges"])


@tool
def dispute_charge(charge_id: str, reason: str, runtime: Runtime) -> str:
    """File a dispute for a wrong charge of the verified customer. `charge_id` from get_recent_charges; `reason` in their words."""
    if not (phone := _phone(runtime)):
        return NOT_VERIFIED
    if not crm.dispute_charge(phone, charge_id, reason):
        return f"No charge {charge_id} on this account. Check the id in get_recent_charges."
    return f"Dispute for {charge_id} is filed; the answer comes within 30 days."


# --- tariffs --------------------------------------------------------------------------------------------------------


@tool
def list_tariffs() -> str:
    """All available tariffs: id, name, monthly price in rubles, data (GB, null is unlimited), minutes, SMS."""
    return _json(crm.tariffs())


@tool
def get_current_tariff(runtime: Runtime) -> str:
    """The verified customer's current tariff."""
    if not (phone := _phone(runtime)):
        return NOT_VERIFIED
    return _json(crm.tariff(crm.customer(phone)["tariff"]))


@tool
def change_tariff(tariff_id: str, runtime: Runtime) -> str:
    """Switch the verified customer to the tariff `tariff_id` (from list_tariffs). The customer confirms it before it runs."""
    if not (phone := _phone(runtime)):
        return NOT_VERIFIED
    if (chosen := crm.tariff(tariff_id)) is None:
        return f"Unknown tariff {tariff_id!r}. Use an id from list_tariffs."
    if crm.customer(phone)["tariff"] == tariff_id:
        return "The customer is already on this tariff."
    crm.change_tariff(phone, tariff_id)
    return f"Done: the tariff is now {chosen['name']}."


# --- tech -----------------------------------------------------------------------------------------------------------


@tool
def check_outages(region: str) -> str:
    """Known network problems in the customer's city or region (e.g. 'Казань')."""
    return _json(crm.outages(region) or "No known problems in this region.")


@tool
def run_line_diagnostics(runtime: Runtime) -> str:
    """Check the verified customer's line: registration in the network, data access, balance."""
    if not (phone := _phone(runtime)):
        return NOT_VERIFIED
    account = crm.customer(phone)
    problems = []
    if account["balance"] < 0:
        problems.append("Negative balance: mobile data and outgoing calls are limited until it is topped up.")
    if outage := crm.outages(account["region"]):
        problems.append(outage[0]["status"])
    return _json({"registered_in_network": True, "problems": problems or ["No problems found on the line."]})


# --- human ----------------------------------------------------------------------------------------------------------


@tool
def add_ticket_note(note: str, runtime: Runtime) -> str:
    """Add what the customer said to the operator's ticket."""
    ticket_id = runtime.state.get("ticket_id")
    if not ticket_id or not crm.add_ticket_note(ticket_id, note):
        return "There is no ticket to add to."
    return f"Added to ticket {ticket_id}."


ALL_TOOLS: list[BaseTool] = [
    *HANDOFF_TOOLS,
    verify_customer,
    search_knowledge_base,
    get_balance,
    get_recent_charges,
    dispute_charge,
    list_tariffs,
    get_current_tariff,
    change_tariff,
    check_outages,
    run_line_diagnostics,
    add_ticket_note,
]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

# Consistency: every tool named in agents.py exists.
_known = set(TOOLS_BY_NAME)
for _agent in AGENTS.values():
    _missing = (set(_agent.tools) | {f"transfer_to_{h}" for h in _agent.handoffs}) - _known
    assert not _missing, f"{_agent.name}: unknown tools {_missing}"
assert ACCOUNT_TOOLS <= _known
