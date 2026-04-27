from prometheus_client import Counter, Gauge, Histogram

WS_CONNECTIONS_ACTIVE = Gauge(
    "router_ws_connections_active",
    "Currently active WebSocket connections",
)

WS_CONNECTIONS_TOTAL = Counter(
    "router_ws_connections_total",
    "Total WebSocket connections accepted",
)

WS_AUTH_FAILURES_TOTAL = Counter(
    "router_ws_auth_failures_total",
    "Total WebSocket authentication failures",
)

WS_MESSAGES_IN_TOTAL = Counter(
    "router_ws_messages_in_total",
    "Total inbound WebSocket messages",
)

WS_MESSAGES_OUT_TOTAL = Counter(
    "router_ws_messages_out_total",
    "Total outbound WebSocket messages",
)

WS_INVALID_MESSAGES_TOTAL = Counter(
    "router_ws_invalid_messages_total",
    "Total invalid inbound WebSocket messages",
)

WS_HEARTBEAT_TIMEOUTS_TOTAL = Counter(
    "router_ws_heartbeat_timeouts_total",
    "Total connections closed due to heartbeat timeout",
)

WS_MESSAGE_PROCESSING_SECONDS = Histogram(
    "router_ws_message_processing_seconds",
    "Processing time for inbound messages",
)