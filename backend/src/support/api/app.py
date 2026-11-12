"""The HTTP API (`support-api`): the support agent over HTTP.

POST   /chats/                          ask, wait for the whole answer
POST   /chats/stream                    ask, server-sent events (status, agent = a handoff, answer, done)
POST   /chats/{id}/decision[/stream]    approve or reject an action the agent paused at (changing the tariff)
POST   /chats/{id}/cancel               stop the question being answered
GET    /chats, /chats/{id}              the client's chats, to come back to one (kept in Postgres)
PATCH  /chats/{id}   DELETE /chats/{id} rename, delete (with the agent's memory of it)
POST   /chats/{id}/answers/{answer_id}/feedback    👍/👎 (an annotation on the trace in Phoenix)
GET    /health
"""

import asyncio
import contextlib
import logging

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from support.agent import tracing
from support.api import chats, runner
from support.db import init, pg
from support.settings import settings

log = logging.getLogger("support.api")


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    tracing.log_status()
    await asyncio.to_thread(init.init)  # the tables of the API (idempotent)
    async with runner.memory():  # the agent with its memory of the chats in Postgres
        yield
    pg.close()
    tracing.flush()


app = FastAPI(
    title="Support Agent API",
    summary="A stateful customer-support agent with handoffs between specialist agents.",
    version="0.1.0",
    root_path=settings.api_root_path,
    redirect_slashes=False,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=settings.api_cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chats.router)


@app.get("/health", tags=["System"])
async def health() -> dict:
    return {"status": "ok"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, forwarded_allow_ips="*")
