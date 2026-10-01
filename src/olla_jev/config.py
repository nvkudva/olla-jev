"""Settings from OLLAJEV_* environment variables, plus the saved config file."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import platformdirs

log = logging.getLogger(__name__)
_lock = threading.RLock()

DEFAULT_MODEL = "Mapika/decider-4b-GGUF:Q4_K_M"
DEFAULT_PORT = 8000


APP = "olla-jev"


def config_dir() -> Path:
    """OLLAJEV_HOME, else ~/.olla-jev."""
    home = os.environ.get("OLLAJEV_HOME")
    return Path(home) if home else Path.home() / ".olla-jev"


def log_dir() -> Path:
    return config_dir() / "logs"


def config_path() -> Path:
    return config_dir() / "config.json"


def host() -> tuple[str, int]:
    """OLLAJEV_HOST as host or host:port, like OLLAMA_HOST. Default 127.0.0.1:8000."""
    value = os.environ.get("OLLAJEV_HOST", "")
    value = value.removeprefix("http://").removeprefix("https://").rstrip("/")
    if not value:
        return "127.0.0.1", DEFAULT_PORT
    if value.startswith("["):  # [::1]:8000
        addr, _, port = value[1:].partition("]")
        return addr, _port(port.lstrip(":") or str(DEFAULT_PORT))
    if value.count(":") == 1:
        addr, port = value.split(":")
        return addr or "127.0.0.1", _port(port)
    return value, DEFAULT_PORT


def _port(text: str) -> int:
    if not text.isdigit() or not 0 < int(text) < 65536:
        raise ValueError(f"OLLAJEV_HOST has an invalid port {text!r}; use host:port, e.g. 127.0.0.1:8000")
    return int(text)


def api_key() -> str | None:
    """OLLAJEV_API_KEY: when set, the server requires `Authorization: Bearer <key>` on every API route."""
    return os.environ.get("OLLAJEV_API_KEY") or None


def is_loopback(addr: str) -> bool:
    return addr in ("localhost", "::1") or addr.startswith("127.")


def models_dir() -> str | None:
    """OLLAJEV_MODELS overrides where weights are stored; None means the shared Hugging Face cache."""
    return os.environ.get("OLLAJEV_MODELS") or None


def keep_alive() -> float:
    """Seconds an idle model stays loaded. OLLAJEV_KEEP_ALIVE accepts 300, 5m, 1h, or -1 for forever."""
    try:
        return parse_duration(os.environ.get("OLLAJEV_KEEP_ALIVE", "5m"))
    except ValueError:
        raise ValueError("OLLAJEV_KEEP_ALIVE must be seconds or a duration like 5m, 1h, -1") from None


def max_loaded_models() -> int:
    try:
        return int(os.environ.get("OLLAJEV_MAX_LOADED_MODELS", "1"))
    except ValueError:
        raise ValueError("OLLAJEV_MAX_LOADED_MODELS must be a whole number") from None


def max_body_bytes() -> int:
    """Largest request body the API accepts. OLLAJEV_MAX_BODY_BYTES overrides the 8 MiB default."""
    try:
        return int(os.environ.get("OLLAJEV_MAX_BODY_BYTES", 8 * 1024 * 1024))
    except ValueError:
        raise ValueError("OLLAJEV_MAX_BODY_BYTES must be a whole number") from None


def device() -> str | None:
    """OLLAJEV_DEVICE, else the device saved by setup, forces cpu, mps or cuda; None picks the best one."""
    value = os.environ.get("OLLAJEV_DEVICE") or load().get("device")
    return None if value in (None, "", "auto") else value


def parse_duration(text: str | float | int) -> float:
    if isinstance(text, (int, float)):
        return float(text)
    text = text.strip()
    units = {"s": 1, "m": 60, "h": 3600}
    if text and text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def _legacy_path() -> Path:
    """Where releases before 0.2 kept config.json: the OS config folder."""
    return Path(platformdirs.user_config_dir(APP, appauthor=False)) / "config.json"


def load() -> dict[str, Any]:
    path = config_path()
    if not path.exists() and "OLLAJEV_HOME" not in os.environ and _legacy_path().exists():
        path = _legacy_path()  # first run after the move to ~/.olla-jev; the next save writes the new file
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        backup = path.with_suffix(".json.bad")
        path.replace(backup)
        log.warning("%s is not valid JSON; moved it to %s and started with empty settings", path, backup)
        return {}


def save(data: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="config.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise


@contextlib.contextmanager
def edit() -> Iterator[dict[str, Any]]:
    """Read, change and save the config as one step, so threads cannot overwrite each other's changes."""
    with _lock:
        data = load()
        yield data
        save(data)


def update(**fields: Any) -> dict[str, Any]:
    with edit() as data:
        data.update(fields)
    return data
