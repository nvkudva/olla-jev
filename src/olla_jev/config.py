"""Settings from OLLAJEV_* environment variables, plus the saved config file."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import platformdirs

DEFAULT_MODEL = "Mapika/decider-4b-GGUF:Q4_K_M"
DEFAULT_PORT = 8000


APP = "olla-jev"


def config_dir() -> Path:
    """OLLAJEV_HOME, else the OS config folder: ~/.config/olla-jev (Linux),
    ~/Library/Application Support/olla-jev (macOS), %APPDATA%\\olla-jev (Windows)."""
    home = os.environ.get("OLLAJEV_HOME")
    return Path(home) if home else Path(platformdirs.user_config_dir(APP, appauthor=False))


def log_dir() -> Path:
    """OLLAJEV_HOME/logs, else the OS log folder: ~/.local/state/olla-jev/log (Linux),
    ~/Library/Logs/olla-jev (macOS)."""
    home = os.environ.get("OLLAJEV_HOME")
    return Path(home) / "logs" if home else Path(platformdirs.user_log_dir(APP, appauthor=False))


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
        return addr, int(port.lstrip(":") or DEFAULT_PORT)
    if value.count(":") == 1:
        addr, port = value.split(":")
        return addr or "127.0.0.1", int(port)
    return value, DEFAULT_PORT


def models_dir() -> str | None:
    """OLLAJEV_MODELS overrides where weights are stored; None means the shared Hugging Face cache."""
    return os.environ.get("OLLAJEV_MODELS") or None


def keep_alive() -> float:
    """Seconds an idle model stays loaded. OLLAJEV_KEEP_ALIVE accepts 300, 5m, 1h, or -1 for forever."""
    return parse_duration(os.environ.get("OLLAJEV_KEEP_ALIVE", "5m"))


def max_loaded_models() -> int:
    return int(os.environ.get("OLLAJEV_MAX_LOADED_MODELS", "1"))


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


def load() -> dict[str, Any]:
    try:
        return json.loads(config_path().read_text())
    except FileNotFoundError:
        return {}


def save(data: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def update(**fields: Any) -> dict[str, Any]:
    data = load()
    data.update(fields)
    save(data)
    return data
