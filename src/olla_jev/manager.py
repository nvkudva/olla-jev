"""Loaded models, Ollama style: load on first request, unload after keep_alive, at most N at once."""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import config, store
from .adapters import pick_device
from .adapters.base import Adapter
from .names import quant_of

log = logging.getLogger(__name__)

# typesafe-sdk's DEFAULT_MODEL and Laya's stock name. Clients that never pass `model=` send one of
# these, so they mean "the default model".
DEFAULT_ALIASES = frozenset({"jev-latest", "laya"})


class NotDownloaded(LookupError):
    pass


class NotTrusted(PermissionError):
    pass


def canonical(r: store.Resolved) -> str:
    """One name per weight file: `repo` or `repo:QUANT`, whatever spelling the request used."""
    if r.gguf is None:
        return r.repo_id
    return f"{r.repo_id}:{quant_of(r.gguf) or r.gguf.rsplit('/', 1)[-1]}"


def default_model() -> str:
    return config.load().get("default_model") or config.DEFAULT_MODEL


def lookup(name: str | None) -> str:
    """Request model name -> the name to resolve, following `cp` aliases and the default aliases."""
    if not name or name in DEFAULT_ALIASES:
        name = default_model()
    return config.load().get("aliases", {}).get(name, name)


@dataclass
class Slot:
    name: str
    adapter: Adapter
    resolved: store.Resolved
    device: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    expires: float = 0.0  # monotonic deadline; inf = never
    pinned: bool = False  # the model `serve` preloaded stays until the server stops
    loaded_at: float = field(default_factory=time.time)


class Manager:
    def __init__(self) -> None:
        self._slots: dict[str, Slot] = {}
        self._guard = threading.RLock()
        self._load_lock = threading.Lock()  # one load at a time: loads are memory spikes
        self._reaper = threading.Thread(target=self._reap, name="olla-jev-reaper", daemon=True)
        self._reaper.start()

    def resolve(self, name: str | None) -> store.Resolved:
        try:
            return store.resolve(lookup(name), online=False)
        except LookupError as exc:
            raise NotDownloaded(str(exc)) from None

    def get(self, name: str | None, keep_alive: float | None = None) -> Slot:
        r = self.resolve(name)
        key = canonical(r)
        with self._guard:
            slot = self._slots.get(key)
        if slot is None:
            with self._load_lock:
                with self._guard:
                    slot = self._slots.get(key)
                if slot is None:
                    slot = self._load(key, r)
        self._touch(slot, keep_alive)
        return slot

    def _load(self, key: str, r: store.Resolved) -> Slot:
        if not store.is_trusted(r):
            raise NotTrusted(
                f"{key} runs Python code from its repo at {r.revision[:12]}; review it, then run: "
                f"olla-jev pull {key} --trust"
            )
        path = store.local_path(r)
        if path is None:
            raise NotDownloaded(f"{key} is not downloaded; run: olla-jev pull {key}")
        while len(self._slots) >= max(1, config.max_loaded_models()):
            self.unload(min(self._slots.values(), key=lambda s: s.expires).name)
        device = pick_device(config.device())
        log.info("loading %s on %s", key, device)
        started = time.monotonic()
        adapter = r.family.load(path, r, device)
        adapter.name = key
        log.info("loaded %s in %.1fs", key, time.monotonic() - started)
        slot = Slot(key, adapter, r, getattr(adapter, "device", None) or device)
        with self._guard:
            self._slots[key] = slot
        return slot

    def _touch(self, slot: Slot, keep_alive: float | None) -> None:
        if keep_alive is not None and keep_alive < 0:
            slot.pinned = True
        if slot.pinned:
            slot.expires = float("inf")
            return
        seconds = config.keep_alive() if keep_alive is None else keep_alive
        slot.expires = float("inf") if seconds < 0 else time.monotonic() + seconds

    def unload(self, name: str) -> bool:
        with self._guard:
            slot = self._slots.pop(name, None)
        if slot is None:
            return False
        with slot.lock:  # let a running request finish
            close = getattr(slot.adapter, "close", None)
            if close:
                close()
        del slot
        gc.collect()
        _empty_device_cache()
        log.info("unloaded %s", name)
        return True

    def unload_all(self) -> None:
        for name in list(self._slots):
            self.unload(name)

    def loaded(self) -> list[Slot]:
        with self._guard:
            return list(self._slots.values())

    def _reap(self) -> None:
        while True:
            time.sleep(1)
            now = time.monotonic()
            for slot in self.loaded():
                if slot.expires <= now and not slot.lock.locked():
                    self.unload(slot.name)

    def run(self, name: str | None, state: Any, questions: dict[str, dict[str, Any]], keep_alive: float | None = None) -> tuple[Slot, dict[str, Any]]:
        slot = self.get(name, keep_alive)
        check_limits(slot.adapter.limits, questions)
        # One forward pass per model at a time: on MPS concurrent forwards abort the process with a
        # Metal command-buffer assertion, and several adapters keep per-call state.
        with slot.lock:
            result = slot.adapter.system_one(state, questions)
        self._touch(slot, keep_alive)
        return slot, result


def check_limits(limits: dict[str, Any], questions: dict[str, dict[str, Any]]) -> None:
    """Reject what the loaded model cannot take before it runs. Token limits stay with the model."""
    if (n := limits.get("max_questions")) and len(questions) > n:
        raise ValueError(f"this model takes at most {n} questions per request, got {len(questions)}")
    for qid, q in questions.items():
        count = len(q.get("criteria") or ())
        if q["type"] == "choice" and (n := limits.get("max_options")) and count > n:
            raise ValueError(f"question {qid!r}: this model takes at most {n} choice options, got {count}")
        if q["type"] == "score" and (n := limits.get("max_levels")) and count > n:
            raise ValueError(f"question {qid!r}: this model takes at most {n} score levels, got {count}")


def _empty_device_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # cache release is best effort
        pass
