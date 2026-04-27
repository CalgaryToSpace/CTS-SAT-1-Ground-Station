import asyncio
import time
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
from pydantic import ValidationError
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from config import (
    logger,
    HEARTBEAT_INTERVAL_SECONDS,
    HEARTBEAT_TIMEOUT_SECONDS,
)
from manager import manager, ClientSession
from schemas import (
    parse_message,
    AuthMessage,
    PingMessage,
    TelemetryMessage,
    AckMessage,
    EventMessage,
)
from auth import authenticate
from metrics import *


async def heartbeat_monitor():
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

        sessions = await manager.snapshot()
        now = time.monotonic()

        for session in sessions:
            if now - session.last_seen > HEARTBEAT_TIMEOUT_SECONDS:
                WS_HEARTBEAT_TIMEOUTS_TOTAL.inc()
                await session.websocket.close(code=1008)
                await manager.remove(session)
                continue

            try:
                await manager.send_json(session, {
                    "type": "heartbeat",
                    "ts": time.time(),
                })
            except Exception:
                await manager.remove(session)


heartbeat_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_task
    heartbeat_task = asyncio.create_task(heartbeat_monitor())
    yield
    heartbeat_task.cancel()


app = FastAPI(title="Secure Dashboard Router", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/publish/{dashboard_id}")
async def publish(dashboard_id: str, payload: dict):
    delivered = await manager.send_to_dashboard(dashboard_id, {
        "type": "router_message",
        "payload": payload,
        "ts": time.time(),
    })
    return {"delivered": delivered}


@app.post("/api/broadcast")
async def broadcast(payload: dict):
    delivered = await manager.broadcast({
        "type": "router_broadcast",
        "payload": payload,
        "ts": time.time(),
    })
    return {"delivered": delivered}


@app.websocket("/ws")
async def websocket_router(websocket: WebSocket):
    await websocket.accept()
    session = ClientSession(websocket)
    await manager.add(session)

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        session.touch()

        first = parse_message(raw)

        if not isinstance(first, AuthMessage):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        ok = await authenticate(session, first.dashboard_id, first.api_key)
        if not ok:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        while True:
            raw = await websocket.receive_text()
            session.touch()

            try:
                msg = parse_message(raw)
            except Exception:
                continue

            if isinstance(msg, PingMessage):
                await manager.send_json(session, {"type": "pong"})

            elif isinstance(msg, TelemetryMessage):
                await manager.send_json(session, {"type": "telemetry_ack"})

            elif isinstance(msg, EventMessage):
                await manager.send_json(session, {"type": "event_ack"})

    except WebSocketDisconnect:
        await manager.remove(session)
    finally:
        await manager.remove(session)