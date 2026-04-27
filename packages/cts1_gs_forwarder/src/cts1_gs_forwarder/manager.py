import asyncio
import uuid

from fastapi import WebSocket

from metrics import (
    WS_CONNECTIONS_ACTIVE,
    WS_CONNECTIONS_TOTAL,
    WS_MESSAGES_OUT_TOTAL,
)
from config import logger


class ClientSession:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.connection_id = str(uuid.uuid4())
        self.dashboard_id = None
        self.authenticated = False
        self.last_seen = 0
        self.connected_at = 0

    def touch(self):
        import time
        self.last_seen = time.monotonic()


class ConnectionManager:
    def __init__(self):
        self._connections = {}
        self._lock = asyncio.Lock()

    async def add(self, session: ClientSession):
        async with self._lock:
            self._connections[session.connection_id] = session
            WS_CONNECTIONS_ACTIVE.inc()
            WS_CONNECTIONS_TOTAL.inc()

    async def remove(self, session: ClientSession):
        async with self._lock:
            if self._connections.pop(session.connection_id, None):
                WS_CONNECTIONS_ACTIVE.dec()

    async def send_json(self, session: ClientSession, payload: dict):
        await session.websocket.send_json(payload)
        WS_MESSAGES_OUT_TOTAL.inc()

    async def send_to_dashboard(self, dashboard_id: str, payload: dict):
        count = 0
        async with self._lock:
            sessions = list(self._connections.values())

        for s in sessions:
            if s.authenticated and s.dashboard_id == dashboard_id:
                try:
                    await s.websocket.send_json(payload)
                    WS_MESSAGES_OUT_TOTAL.inc()
                    count += 1
                except Exception:
                    logger.exception("send failure")
        return count

    async def broadcast(self, payload: dict):
        count = 0
        async with self._lock:
            sessions = list(self._connections.values())

        for s in sessions:
            if s.authenticated:
                try:
                    await s.websocket.send_json(payload)
                    WS_MESSAGES_OUT_TOTAL.inc()
                    count += 1
                except Exception:
                    logger.exception("broadcast failure")
        return count

    async def snapshot(self):
        async with self._lock:
            return list(self._connections.values())


manager = ConnectionManager()