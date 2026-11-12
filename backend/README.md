# Support Agent: backend

Шаблон для хакатона: агент поддержки оператора связи с **состоянием** и **handoff-архитектурой**. Стек перенесён из cAtIx без изменений: Python 3.13 + uv, FastAPI + uvicorn, LangChain `create_agent` / LangGraph (OpenAI-совместимая LLM, например llama.cpp), Postgres (чекпоинты `AsyncPostgresSaver` + таблицы чатов), Arize Phoenix, pytest, ruff/prek, Docker.

Убрано относительно cAtIx: MCP, Telegram-бот и userbot'ы (Telegram, MAX), Redis, воркер, docs/wiki-парсеры, вложения, календарь, Moodle, поиск и эмбеддинги.

## Как это устроено

```
сообщение ─► guardrail ─► модель, как АКТИВНЫЙ специалист (его промпт и его инструменты)
                            │ transfer_to_<x> ─► active_agent = x, цикл продолжает уже x
                            │ change_tariff   ─► пауза: клиент подтверждает действие (HITL)
                            └ текст           ─► ответ; состояние сохраняется в Postgres
```

Один агент LangGraph, у которого в состоянии (`agent/state.py`) лежит `active_agent`, `verified`, `ticket_id`, `escalated`.
`HandoffMiddleware` (`agent/core.py`) перед каждым вызовом модели подставляет промпт и набор инструментов активного специалиста;
handoff это инструмент `transfer_to_<имя>` (`agent/tools.py`), возвращающий `Command(update={"active_agent": ...})`.
Состояние чекпоинтится, поэтому следующее сообщение чата попадает к тому же специалисту.

| Агент | Что делает | Куда может передать |
|---|---|---|
| `triage` | встречает, отвечает на простое из базы знаний, маршрутизирует | billing, tariffs, tech, human |
| `billing` | баланс, списания, спор по списанию | triage, tariffs, tech, human |
| `tariffs` | сравнение и смена тарифа (с подтверждением клиента) | triage, billing, human |
| `tech` | сбои на сети, диагностика линии | triage, human |
| `human` | заводит обращение, оператор «подхватывает» (терминальное состояние) | |

Что проверяется кодом, а не промптом:
- граф разрешённых передач (`agents.py`); недопустимый `transfer_to_*` ничего не меняет;
- данные аккаунта доступны только после `verify_customer` (телефон + PIN): до этого модель не видит account-инструментов, а сами инструменты отказывают; после 3 неверных PIN верификация блокируется;
- `change_tariff` выполняется только после `approve` клиента (`HumanInTheLoopMiddleware`; ответ с `approval`, затем `POST /chats/{id}/decision`);
- нет параллельных вызовов инструментов (две передачи одновременно конфликтовали бы).

Демо-данные: `src/support/data/` и `src/support/crm.py` (мок бэк-офиса в памяти, перезапуск сбрасывает его). Демо-клиенты: `+7 999 000-11-22` PIN `1234`; `+7 999 000-33-44` PIN `4321`.
**Свой проект:** замените `crm.py` вызовами реальных систем, поменяйте специалистов в `agent/agents.py`, их инструменты в `agent/tools.py`.

## Запуск

```bash
cp .env.example .env              # LLM_ROUTE: адрес OpenAI-совместимого сервера
docker compose up -d db phoenix   # Postgres и трассировки (http://localhost:6006)
uv sync
uv run support-api                # http://localhost:8000/docs; таблицы создаются при старте
uv run support "у меня не работает интернет"   # или `uv run support` для диалога в терминале (без БД)
```

Всё в Docker: `docker compose --profile api up -d --build`.

## API

Идентификация: заголовок `X-Client-Id` (8–64 символа `A-Za-z0-9_-`, фронтенд генерирует один раз). Это не аутентификация: замените `api/auth.py`.

| Метод | Путь | |
|---|---|---|
| POST | `/chats/stream` | вопрос → SSE: `chat`, `status`, `agent` (handoff), `answer` (весь текст на данный момент), `done` / `error` / `cancelled` |
| POST | `/chats/` | вопрос → готовый `Answer` |
| POST | `/chats/{id}/decision[/stream]` | `approve` / `reject` действия, на котором агент остановился |
| POST | `/chats/{id}/cancel` | прервать вопрос |
| GET | `/chats`, `/chats/{id}` | чаты клиента, история чата (с состоянием и ожидающим подтверждения) |
| PATCH / DELETE | `/chats/{id}` | переименовать / удалить вместе с памятью агента |
| POST | `/chats/{id}/answers/{answer_id}/feedback` | 👍/👎 (аннотация к трассе в Phoenix) |

`Answer`: `answer`, `state` (`active_agent`, `verified`, `customer_name`, `ticket_id`, `escalated`), `handoffs` за этот ход, `approval`.

## Тесты и проверки

```bash
uv run pytest                                  # логика handoff-ов на сценарной модели, без LLM и БД
TEST_DATABASE_URL=postgresql://support:support@127.0.0.1:5432/support uv run pytest   # + API на Postgres
uv run ruff check . && uv run ruff format --check .
```

Модель в тестах сценарная (`tests/helpers.py`): тест решает, что она «скажет», и читает, какой промпт и какие инструменты она получила.
