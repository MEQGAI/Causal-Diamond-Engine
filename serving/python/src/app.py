"""Minimal ASGI placeholder for the Python serving runtime."""

from __future__ import annotations

import json
from typing import Awaitable, Callable

ASGIReceive = Callable[[], Awaitable[dict]]
ASGISend = Callable[[dict], Awaitable[None]]


async def app(scope: dict, receive: ASGIReceive, send: ASGISend) -> None:
    if scope["type"] != "http":  # pragma: no cover - uvicorn never exercises this in --help
        return

    body = json.dumps({"status": "ok", "service": "foundation-serving"}).encode()
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [(b"content-type", b"application/json")],
    })
    await send({"type": "http.response.body", "body": body})
