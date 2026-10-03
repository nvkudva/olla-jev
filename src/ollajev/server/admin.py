"""Ollama-style model management: /api/tags, /api/ps, /api/pull, /api/show, /api/delete, /api/copy, /api/stop."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .. import config, names, store
from ..catalog import CATALOG
from ..manager import canonical, lookup

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
_pulling: set[str] = set()
_pulling_guard = threading.Lock()
DESCRIPTIONS = {e.name: e.description for e in CATALOG}


class ModelRef(BaseModel):
    model: str


class PullRequest(BaseModel):
    model: str
    stream: bool = True


class CopyRequest(BaseModel):
    source: str
    destination: str


def _describe(name: str, resolved: store.Resolved, size: int, modified: float) -> dict[str, Any]:
    return {
        "name": name,
        "model": name,
        "size": size,
        "modified_at": datetime.fromtimestamp(modified, UTC).isoformat(),
        "digest": resolved.revision,
        "description": DESCRIPTIONS.get(name, f"{resolved.family.name} typed-decision model"),
        # TypeSafe clients require a string; fall back to the download date for repos pinned offline.
        "release_date": store.released(resolved.repo_id) or datetime.fromtimestamp(modified, UTC).date().isoformat(),
        "details": {
            "family": resolved.family.name,
            "format": names.format_of(resolved.weights) if resolved.weights else "safetensors",
            "quantization_level": names.tag_of(resolved.weights) if resolved.weights else None,
        },
        "limits": resolved.family.limits(resolved),
    }


def tags() -> list[dict[str, Any]]:
    """One entry per downloaded weight file: a GGUF repo with two quants on disk is two models, and so is an ONNX
    repo with two exports."""
    out = []
    for repo_id, (size, modified) in sorted(store.downloaded().items()):
        revision = store.pins()[repo_id]
        rev = store.snapshot(repo_id, revision)
        if rev is None:
            continue
        snap = Path(rev.snapshot_path)
        local = names.labels(sorted(str(p.relative_to(snap)) for p in snap.glob("**/*")))
        for name in [f"{repo_id}:{tag}" for tag in local.values()] or [repo_id]:
            try:
                resolved = store.resolve(name, online=False)
            except (LookupError, ValueError):
                continue
            if resolved.weights:
                weight = sum(
                    (snap / f).stat().st_size
                    for f in [resolved.weights, *names.sidecars(resolved.weights)]
                    if (snap / f).is_file()
                )
            else:
                weight = size
            out.append(_describe(canonical(resolved), resolved, weight, modified))
    return out


@router.get("/tags")
def api_tags() -> dict[str, Any]:
    return {"models": tags()}


@router.get("/ps")
def api_ps() -> dict[str, Any]:
    from . import api

    now = time.monotonic()
    models = []
    for slot in api.current_manager().loaded():
        expires = (
            None
            if slot.expires == float("inf")
            else datetime.fromtimestamp(datetime.now().timestamp() + (slot.expires - now), UTC).isoformat()
        )
        models.append(
            {
                "name": slot.name,
                "model": slot.name,
                "device": slot.device,
                "expires_at": expires,
                "details": {"family": slot.resolved.family.name},
            }
        )
    return {"models": models}


@router.post("/show")
def api_show(req: ModelRef) -> Any:
    try:
        resolved = store.resolve(lookup(req.model), online=False)
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    return {
        "model": canonical(resolved),
        "repo": resolved.repo_id,
        "revision": resolved.revision,
        "file": resolved.weights,
        "family": resolved.family.name,
        "runs_repo_code": resolved.family.runs_repo_code,
        "trusted": store.is_trusted(resolved),
        "release_date": store.released(resolved.repo_id),
        "limits": resolved.family.limits(resolved),
        "path": store.local_path(resolved),
    }


def report_bytes(resolved: store.Resolved, status: str, events: queue.Queue[dict[str, Any] | None]) -> None:
    """Download `r`, putting a `completed`/`total` byte event on `events` as the cache grows."""
    done = threading.Event()
    try:
        total = store.download_size(resolved)
    except Exception:  # progress is optional; the download itself reports real errors
        total = 0
    start = store.bytes_on_disk(resolved.repo_id)

    def poll() -> None:
        last = -1
        while not done.wait(0.5):
            completed = min(store.bytes_on_disk(resolved.repo_id) - start, total)
            if completed != last:
                last = completed
                events.put({"status": status, "digest": resolved.revision, "total": total, "completed": completed})

    if total:
        threading.Thread(target=poll, daemon=True).start()
    try:
        store.download(resolved)
    finally:
        done.set()


@router.post("/pull")
def api_pull(req: PullRequest) -> Any:
    """Download a model; with stream=true, NDJSON status lines like Ollama's pull."""
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def work() -> None:
        key = None
        try:
            events.put({"status": "pulling manifest"})
            resolved = store.resolve(lookup(req.model))
            if resolved.family.runs_repo_code and not store.is_trusted(resolved):
                events.put(
                    {
                        "error": f"{canonical(resolved)} runs Python code from its repo; trust is not available over HTTP. "
                        f"Review https://huggingface.co/{resolved.repo_id}/tree/{resolved.revision}, then run: "
                        f"ollajev pull {canonical(resolved)} --trust"
                    }
                )
                return
            key = resolved.repo_id
            with _pulling_guard:
                busy = key in _pulling
                _pulling.add(key)
            if busy:
                key = None
                events.put({"error": f"{resolved.repo_id} is already being pulled"})
                return
            status = f"downloading {resolved.repo_id}@{resolved.revision[:12]}"
            events.put({"status": status, "digest": resolved.revision})
            report_bytes(resolved, status, events)
            prefetch = getattr(resolved.family, "prefetch", None)
            if prefetch:
                events.put({"status": "downloading base model"})
                prefetch(store.local_path(resolved))
            events.put({"status": "success", "model": canonical(resolved)})
        except (LookupError, ValueError) as exc:  # bad name or unknown model: safe to show
            events.put({"error": str(exc)})
        except Exception:
            log.exception("pull of %s failed", req.model)
            events.put({"error": "pull failed; see the server log"})
        finally:
            if key:
                with _pulling_guard:
                    _pulling.discard(key)
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def lines():
        while (event := events.get()) is not None:
            yield json.dumps(event) + "\n"

    if req.stream:
        return StreamingResponse(lines(), media_type="application/x-ndjson")
    last = [json.loads(line) for line in lines()][-1]
    return JSONResponse(status_code=400 if "error" in last else 200, content=last)


@router.delete("/delete")
def api_delete(req: ModelRef) -> Any:
    from . import api

    with config.edit() as data:
        removed = data.get("aliases", {}).pop(req.model, None)
    if removed is not None:
        return {"status": "success"}
    try:
        resolved = store.resolve(lookup(req.model), online=False)
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    api.current_manager().unload(canonical(resolved))
    return {"status": "success", "freed": store.remove(resolved)}


@router.post("/copy")
def api_copy(req: CopyRequest) -> Any:
    target = lookup(req.source)
    with config.edit() as data:
        data.setdefault("aliases", {})[req.destination] = target
    return {"status": "success"}


@router.post("/stop")
def api_stop(req: ModelRef) -> Any:
    from . import api

    try:
        resolved = store.resolve(lookup(req.model), online=False)
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    return {"status": "success" if api.current_manager().unload(canonical(resolved)) else "not loaded"}
