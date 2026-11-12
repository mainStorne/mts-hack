# Support Agent: шаблон для хакатона

Агент поддержки оператора связи: специалисты (triage, billing, tariffs, tech, human) передают друг другу диалог (handoff), состояние диалога хранится в Postgres.

- [`backend/`](backend/README.md): FastAPI + LangChain/LangGraph + Postgres + Phoenix (стек cAtIx без MCP, Telegram/MAX, Redis, воркера и docs).
- [`frontend/`](frontend/README.md): React 19 + Vite + TanStack + Tailwind/DaisyUI (стек сайта InNoHassle, только интерфейс чата с агентом).

```bash
cd backend && cp .env.example .env && docker compose up -d db phoenix && uv run support-api
cd frontend && pnpm install && pnpm dev        # VITE_API_URL=http://localhost:8000
```
