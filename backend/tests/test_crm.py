from support import crm


def test_verification_normalizes_the_phone_and_checks_the_pin():
    assert crm.verify("8 999 000-11-22", "1234")["name"] == "Анна Петрова"
    assert crm.verify("+79990001122", "0000") is None
    assert crm.verify("+70000000000", "1234") is None


def test_the_customer_is_read_with_plain_numbers_and_dates():
    anna = crm.customer("+79990001122")
    assert anna["tariff"] == "smart" and anna["balance"] == 312.5
    assert [c["date"] for c in anna["charges"]] == ["2026-11-01", "2026-11-05", "2026-11-09"]
    assert crm.tariff("max")["data_gb"] is None  # unlimited


def test_changes_survive_a_seed_and_a_reset_brings_the_seed_back():
    crm.change_tariff("+79990001122", "max")
    assert crm.dispute_charge("+79990001122", "ch-1002", "не подключал")
    assert not crm.dispute_charge("+79990003344", "ch-1002", "чужое списание")  # another customer's charge
    ticket = crm.create_ticket("+79990001122", "нет связи")
    assert crm.add_ticket_note(ticket.id, "после 18:00") and crm.ticket(ticket.id).notes == ["после 18:00"]

    crm.seed()  # what is there is never overwritten
    anna = crm.customer("+79990001122")
    assert anna["tariff"] == "max" and [c["disputed"] for c in anna["charges"]] == [None, "не подключал", None]
    assert crm.ticket(ticket.id) is not None

    crm.reset()
    assert crm.customer("+79990001122")["tariff"] == "smart" and crm.ticket(ticket.id) is None
