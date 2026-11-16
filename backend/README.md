# Support Agent: backend

Шаблон для хакатона: агент поддержки оператора связи с **состоянием** и **handoff-архитектурой**. Стек: Python 3.13 + uv, FastAPI + uvicorn, LangChain `create_agent` / LangGraph (OpenAI-совместимая LLM, например llama.cpp), Postgres (чекпоинты `AsyncPostgresSaver` + таблицы чатов), Arize Phoenix, pytest, ruff/prek, Docker.

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

Демо-бэк-офис (тарифы, клиенты со списаниями, сбои сети, обращения) тоже таблицы Postgres (`crm.py`). Стартовые данные берутся из `data/seed.json` и вставляются, только если их ещё нет (`support-db init` и старт API); смена тарифа, споры и обращения сохраняются между перезапусками. `uv run support-db reset-demo` возвращает всё к `seed.json`.

**База знаний** лежит в Postgres, таблица `kb_articles` (`db/kb.py`), поиск идёт только по ней (`retrieval.py`). Стартовые статьи берутся из `data/knowledge_base.json` и один раз вставляются в таблицу (`support-db init` и старт API); существующие статьи не перезаписываются, дальше источник правды таблица. С `EMBEDDINGS_ROUTE` статьи получают векторы (pgvector), поиск гибридный: полнотекстовый Postgres (русская морфология) и ближайшие по вектору, слияние по рангам (RRF) → реранкер (`RERANKER_ROUTE`) → отсечка по `RERANKER_MIN_SCORE`. Слова находят точные термины (название тарифа, APN, 0890), векторы находят перефразировки. `EMBEDDINGS_ROUTE` обязателен: другого поиска нет, и недоступный сервер эмбеддингов или реранкера это ошибка инструмента (агент сообщит клиенту, что не может проверить базу). Нужно расширение `vector` (образ ParadeDB в compose его содержит, для тестов `pgvector/pgvector:pg17`). Демо-клиенты: `+7 999 000-11-22` PIN `1234`; `+7 999 000-33-44` PIN `4321`.
**Свой проект:** замените `crm.py` (и таблицы бэк-офиса) вызовами реальных систем, поменяйте специалистов в `agent/agents.py`, их инструменты в `agent/tools.py`.

## Запуск

```bash
cp .env.example .env              # LLM_ROUTE: адрес OpenAI-совместимого сервера
docker compose up -d db phoenix   # Postgres и трассировки (http://localhost:6006)
uv sync
uv run support-api                # http://localhost:8000/docs; таблицы создаются при старте
uv run support "у меня не работает интернет"   # или `uv run support` для диалога в терминале (нужна БД с базой знаний)
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
TEST_DATABASE_URL=postgresql://support:support@127.0.0.1:5432/support uv run pytest   # Postgres с pgvector; без него тесты пропускаются
uv run ruff check . && uv run ruff format --check .
```

Модель в тестах сценарная (`tests/helpers.py`): модель без LLM: тест решает, что она «скажет», и читает, какой промпт и какие инструменты она получила.
