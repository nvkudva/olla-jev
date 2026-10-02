"""Talking to a running ollajev server over HTTP, for the front ends."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from . import config


def server_url() -> str:
    """OLLAJEV_HOST when set, else the address the last `serve` bound, else the default."""
    if os.environ.get("OLLAJEV_HOST"):
        host, port = config.host()
        return f"http://{url_host(host)}:{port}"
    return config.load().get("server_url") or f"http://127.0.0.1:{config.DEFAULT_PORT}"


def auth_headers() -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if key := config.api_key():
        headers["authorization"] = f"Bearer {key}"
    return headers


def call(method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 600) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(server_url() + path, data=data, method=method, headers=auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read() or b"{}")
        raise SystemExit(f"error: {payload.get('error') or payload.get('detail') or exc}") from None


def server_running() -> bool:
    try:
        urllib.request.urlopen(server_url() + "/", timeout=0.5).close()
        return True
    except (urllib.error.URLError, OSError):
        return False


def need_server() -> None:
    if not server_running():
        raise SystemExit(f"could not connect to ollajev at {server_url()}; start it with: ollajev serve")


def url_host(host: str) -> str:
    probe = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"[{probe}]" if ":" in probe else probe
