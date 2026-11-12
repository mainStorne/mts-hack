"""What the customer is told while the agent works on a question: the API sends it as `status` events."""

CHECKING = "🛡 Проверяю запрос…"
WRITING = "✍️ Формулирую ответ…"
# The status of a tool while it runs, between the model deciding to call it and the answer starting.
STATUS = {
    "verify_customer": "🔐 Проверяю данные…",
    "search_knowledge_base": "📚 Ищу в базе знаний…",
    "get_balance": "💰 Смотрю баланс…",
    "get_recent_charges": "🧾 Смотрю списания…",
    "dispute_charge": "📝 Оформляю оспаривание…",
    "list_tariffs": "📋 Смотрю тарифы…",
    "get_current_tariff": "📋 Смотрю ваш тариф…",
    "change_tariff": "🔁 Готовлю смену тарифа…",
    "check_outages": "📡 Проверяю сбои на сети…",
    "run_line_diagnostics": "🛠 Диагностирую линию…",
    "add_ticket_note": "📝 Добавляю в обращение…",
    "transfer_to_triage": "↪️ Передаю диалог…",
    "transfer_to_billing": "↪️ Передаю специалисту по оплате…",
    "transfer_to_tariffs": "↪️ Передаю специалисту по тарифам…",
    "transfer_to_tech": "↪️ Передаю техподдержке…",
    "transfer_to_human": "👤 Подключаю оператора…",
}
