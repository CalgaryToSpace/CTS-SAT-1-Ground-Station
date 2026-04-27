import json
import time
from typing import Literal
from pydantic import BaseModel, Field

from config import MAX_MESSAGE_SIZE_BYTES


class BaseEnvelope(BaseModel):
    type: Literal["auth", "ping", "telemetry", "command_ack", "event"]
    ts: float = Field(default_factory=lambda: time.time())


class AuthMessage(BaseEnvelope):
    type: Literal["auth"] = "auth"
    dashboard_id: str
    api_key: str


class PingMessage(BaseEnvelope):
    type: Literal["ping"] = "ping"
    nonce: str | None = None


class TelemetryPayload(BaseModel):
    metric: str
    value: float
    unit: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class TelemetryMessage(BaseEnvelope):
    type: Literal["telemetry"] = "telemetry"
    payload: TelemetryPayload


class AckMessage(BaseEnvelope):
    type: Literal["command_ack"] = "command_ack"
    command_id: str
    status: Literal["ok", "error"]
    detail: str | None = None


class EventPayload(BaseModel):
    name: str
    severity: Literal["info", "warning", "error"]
    data: dict = Field(default_factory=dict)


class EventMessage(BaseEnvelope):
    type: Literal["event"] = "event"
    payload: EventPayload


def parse_message(raw: str):
    if len(raw.encode("utf-8")) > MAX_MESSAGE_SIZE_BYTES:
        raise ValueError("message too large")

    obj = json.loads(raw)
    msg_type = obj.get("type")

    if msg_type == "auth":
        return AuthMessage.model_validate(obj)
    if msg_type == "ping":
        return PingMessage.model_validate(obj)
    if msg_type == "telemetry":
        return TelemetryMessage.model_validate(obj)
    if msg_type == "command_ack":
        return AckMessage.model_validate(obj)
    if msg_type == "event":
        return EventMessage.model_validate(obj)

    raise ValueError(f"unsupported message type: {msg_type}")