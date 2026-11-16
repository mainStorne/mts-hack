"""FastAPI-сервис CFE: транскрипт с метаданными -> факты в схеме кейса.

Логика обработки — в cfe/runtime.py (полный список переменных окружения там).
Запуск (из ml/): CFE_MODEL=runs/struct-lora uv run uvicorn cfe.api:app --host 127.0.0.1 --port 8000
Переменные: CFE_MODEL (каталог LoRA-адаптера с best/ или final/), CFE_PARAMS (путь к params.json из struct_predict: tau_attr и snap_th),
CFE_TAU (общий порог confidence value, 0.8, если нет CFE_PARAMS), CFE_SNAP (порог привязки value к каталогу, 0.5),
CFE_DENOISE=1 (нормализатор ASR-шума; только для модели, обученной на нормализованных данных), CFE_SCHEMA (entities|fact).
Ансамбль GLiNER + Gemma (runs/ens-v1): CFE_ENSEMBLE=<params.json ансамбля>, CFE_GEMMA_MODEL=<каталог базовой Gemma>,
CFE_GEMMA_ADAPTER=<каталог LoRA-адаптера Gemma; пусто — CFE_GEMMA_MODEL уже слитая>. Параметр `tau` запроса в этом режиме не используется.
CFE_GEMMA_BATCH (16) — макс. звонков в одном батче Gemma; ручка /extract_batch принимает до 64 звонков.
"""
import os
import queue
import sys
import threading
from concurrent.futures import Future
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Literal, Optional

from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime import Runtime  # noqa: E402

rt = None


class Turn(BaseModel):
    idx: Optional[int] = None
    speaker: Literal["operator", "client"]
    text: str


class CallIn(BaseModel):
    """Звонок: метаданные и реплики. Остальные поля формата кейса (series, noise_level, …) принимаются и игнорируются."""
    model_config = ConfigDict(extra="ignore")
    call_id: Optional[str] = None
    started_at: str = Field(description="ISO-дата/время начала звонка — якорь относительных дат")
    duration_sec: Optional[int] = None
    turns: list[Turn] = Field(min_length=1)

    @field_validator("started_at")
    @classmethod
    def _iso(cls, v):
        datetime.fromisoformat(v)
        return v


class Obj(BaseModel):
    type: Literal["service", "device", "ticket", "client", "other"]
    name: str


class Source(BaseModel):
    turn_idx: int
    span: Optional[list[int]] = None
    quote: Optional[str] = None


class Fact(BaseModel):
    object: Obj
    attribute: Literal["service_state", "tariff", "ticket_status", "action", "agreement"]
    value: str
    polarity: Literal[1, -1]
    speaker: Literal["operator", "client"]
    valid_from: str
    valid_to: Optional[str] = None
    source: Source


class CallOut(BaseModel):
    call_id: Optional[str]
    facts: list[Fact]


jobs = queue.Queue()  # (call, tau, Future) | None; единственный поток-обработчик владеет GPU


def _worker():
    while True:
        first = jobs.get()
        if first is None:
            return
        batch, stop = [first], False
        while len(batch) < rt.gemma_batch:
            try:
                j = jobs.get_nowait()
            except queue.Empty:
                break
            if j is None:
                stop = True
                break
            batch.append(j)
        try:
            res = rt.process([b[0] for b in batch], [b[1] for b in batch])
            for b, r in zip(batch, res):
                b[2].set_result(r)
        except Exception as e:  # noqa: BLE001
            for b in batch:
                b[2].set_exception(e)
        if stop:
            return


@asynccontextmanager
async def lifespan(_app):
    global rt
    rt = Runtime()
    threading.Thread(target=_worker, daemon=True).start()
    yield
    jobs.put(None)


app = FastAPI(title="CFE: извлечение фактов из транскрипта", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": rt.model_path, "tau": rt.tau, "snap": rt.snap, "ensemble": rt.ens is not None,
            "gemma_batch": rt.gemma_batch, "device": rt.device, "gemma_backend": rt.backend}


def _submit(body: CallIn, tau):
    # внутри пайплайна реплики индексируются позицией; исходные idx возвращаем в source.turn_idx
    orig = [t.idx if t.idx is not None else i for i, t in enumerate(body.turns)]
    call = {"call_id": body.call_id or "call", "started_at": body.started_at,
            "turns": [{"idx": i, "speaker": t.speaker, "text": t.text} for i, t in enumerate(body.turns)]}
    fut = Future()
    jobs.put((call, tau, fut))
    return orig, fut


def _finish(body: CallIn, orig, fut):
    facts = fut.result()
    for f in facts:
        f["source"]["turn_idx"] = orig[f["source"]["turn_idx"]]
    return {"call_id": body.call_id, "facts": facts}


@app.post("/extract", response_model=CallOut)
def extract(body: CallIn, tau: Optional[float] = None):
    """Факты звонка. `tau` — порог детекции (по умолчанию CFE_TAU)."""
    orig, fut = _submit(body, tau)
    return _finish(body, orig, fut)


@app.post("/extract_batch", response_model=list[CallOut])
def extract_batch(body: Annotated[list[CallIn], Body(min_length=1, max_length=64)], tau: Optional[float] = None):
    """Факты для нескольких звонков (до 64); порядок ответа равен порядку входа."""
    subs = [(b, *_submit(b, tau)) for b in body]
    return [_finish(b, orig, fut) for b, orig, fut in subs]
