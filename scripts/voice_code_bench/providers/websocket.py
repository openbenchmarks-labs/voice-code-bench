from __future__ import annotations

import json
from typing import Any

WEBSOCKET_CLOSE_TIMEOUT_SECONDS = 1.0


def websocket_client(feature_name: str) -> Any:
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError(f"Install websocket-client to use {feature_name}.") from exc
    return websocket


def create_websocket(url: str, headers: list[str], feature_name: str, *, timeout: float = 120.0) -> tuple[Any, Any]:
    websocket = websocket_client(feature_name)
    return websocket.create_connection(url, header=headers, timeout=timeout), websocket


def iter_websocket_json_until_closed(ws: Any, websocket: Any) -> Any:
    while True:
        try:
            message = ws.recv()
        except (websocket.WebSocketConnectionClosedException, websocket.WebSocketTimeoutException):
            break
        if not message:
            break
        yield json.loads(message)


def close_websocket_quietly(ws: Any) -> None:
    try:
        ws.settimeout(WEBSOCKET_CLOSE_TIMEOUT_SECONDS)
        ws.close(timeout=WEBSOCKET_CLOSE_TIMEOUT_SECONDS)
    except Exception:
        pass
