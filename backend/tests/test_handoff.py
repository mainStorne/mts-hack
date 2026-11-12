import asyncio

from langgraph.types import Command

from support import crm
from support.agent.core import build_agent, final_answer, session_state, turn_handoffs
from support.agent.tools import MAX_VERIFY_FAILURES
from tests.helpers import ScriptedModel, call, say, use

ANNA = {"phone": "+79990001122", "pin": "1234"}


def run(agent, text, thread="t", **state):
    config = {"configurable": {"thread_id": thread}}
    return asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": text}], **state}, config))


def make(*script):
    model = ScriptedModel(script=list(script))
    return model, build_agent(model, guardrail=False)


def test_handoff_changes_the_speaker_and_the_next_message_goes_to_it():
    model, agent = make(
        use(call("transfer_to_billing", reason="спрашивает про списание")),
        say("Назовите номер телефона и PIN."),
        say("Спасибо, жду данные."),
    )
    result = run(agent, "Почему списали 199 рублей?")
    assert session_state(result)["active_agent"] == "billing"
    assert turn_handoffs(result["messages"]) == [
        {"from_agent": "triage", "to_agent": "billing", "reason": "спрашивает про списание"}
    ]
    assert final_answer(result["messages"]) == "Назовите номер телефона и PIN."
    first, second = model.seen  # the first call is triage, the one after the handoff is billing
    assert "Support specialist" in first["system"] and "billing specialist" in second["system"]

    result = run(agent, "Мой номер +79990001122")  # a new message, the same chat: billing answers
    assert "billing specialist" in model.seen[-1]["system"]
    assert turn_handoffs(result["messages"]) == []  # only this turn's handoffs


def test_account_tools_appear_only_after_verification():
    model, agent = make(
        use(call("transfer_to_billing", reason="баланс")),
        use(call("verify_customer", phone="+79990001122", pin="0000")),
        say("PIN неверный."),
    )
    result = run(agent, "Какой у меня баланс?")
    billing_before = model.seen[1]["tools"]
    assert "get_balance" not in billing_before and "verify_customer" in billing_before
    assert session_state(result)["verified"] is False and result["verify_failures"] == 1

    model.script += [use(call("verify_customer", **ANNA)), use(call("get_balance")), say("На счёте 312.5 ₽.")]
    result = run(agent, "PIN 1234")
    assert session_state(result)["verified"] and session_state(result)["customer_name"] == "Анна Петрова"
    after = [s["tools"] for s in model.seen[3:]]
    assert all("get_balance" in tools and "verify_customer" not in tools for tools in after[1:])
    assert "312.5" in result["messages"][-2].content  # the tool result came from the account of the verified customer


def test_verification_is_blocked_after_repeated_failures():
    model, agent = make(
        *[use(call("verify_customer", phone="+79990001122", pin=str(i))) for i in range(MAX_VERIFY_FAILURES + 1)]
    )
    model.script.append(say("Не получилось."))
    result = run(agent, "проверь меня", active_agent="billing")
    assert session_state(result)["verified"] is False
    assert "blocked" in next(m.content for m in reversed(result["messages"]) if m.type == "tool")


def test_unverified_customer_cannot_read_an_account_even_if_the_model_asks():
    _, agent = make(use(call("get_balance")), say("Нужна проверка."))
    result = run(agent, "баланс", active_agent="billing")
    assert "not verified" in next(m.content for m in result["messages"] if m.type == "tool")


def test_a_handoff_that_the_graph_does_not_allow_changes_nothing():
    _, agent = make(use(call("transfer_to_billing", reason="не по адресу")), say("Продолжаю сам."))
    result = run(agent, "нет интернета", active_agent="tech")
    assert session_state(result)["active_agent"] == "tech"
    assert turn_handoffs(result["messages"]) == []


def test_escalation_creates_one_ticket_and_the_human_agent_stays():
    model, agent = make(
        use(call("transfer_to_human", reason="клиент просит оператора")),
        say("Оператор свяжется с вами."),
        use(call("add_ticket_note", note="удобно после 18:00")),
        say("Записал."),
    )
    result = run(agent, "Хочу поговорить с человеком")
    state = session_state(result)
    assert state["escalated"] and state["active_agent"] == "human" and state["ticket_id"].startswith("T-")
    assert "human operator" in model.seen[1]["system"] and model.seen[1]["tools"] == ["add_ticket_note"]
    assert crm.ticket(state["ticket_id"]).summary == "клиент просит оператора"

    run(agent, "Звоните после 18:00")
    assert crm.ticket(state["ticket_id"]).notes == ["удобно после 18:00"]


def _tariff_change(agent, decision=None):
    state = {"active_agent": "tariffs", "verified": True, "customer_phone": ANNA["phone"], "customer_name": "Анна"}
    result = run(agent, "Хочу тариф Макс", thread="approval", **state)
    config = {"configurable": {"thread_id": "approval"}}
    snapshot = asyncio.run(agent.aget_state(config))
    assert [i.value["action_requests"][0]["name"] for t in snapshot.tasks for i in t.interrupts] == ["change_tariff"]
    assert crm.customer(ANNA["phone"])["tariff"] == "smart"  # nothing changed while it waits
    interrupt = snapshot.tasks[0].interrupts[0]
    assert "Макс" in interrupt.value["action_requests"][0]["description"]
    resume = Command(resume={interrupt.id: {"decisions": [decision]}})
    return asyncio.run(agent.ainvoke(resume, config)), result


def test_changing_the_tariff_waits_for_approval_and_runs_after_it():
    _, agent = make(use(call("change_tariff", tariff_id="max")), say("Готово, вы на тарифе «Макс»."))
    result, _ = _tariff_change(agent, {"type": "approve"})
    assert crm.customer(ANNA["phone"])["tariff"] == "max"
    assert final_answer(result["messages"]).startswith("Готово")


def test_a_rejected_tariff_change_is_not_made():
    _, agent = make(use(call("change_tariff", tariff_id="max")), say("Хорошо, тариф остался прежним."))
    _tariff_change(agent, {"type": "reject", "message": "The customer declined"})
    assert crm.customer(ANNA["phone"])["tariff"] == "smart"
