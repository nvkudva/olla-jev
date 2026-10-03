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

from .. import config, store
from ..catalog import CATALOG
from ..manager import canonical, lookup
from ..names import quant_of

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


def _describe(name: str, r: store.Resolved, size: int, modified: float) -> dict[str, Any]:
    return {
        "name": name,
        "model": name,
        "size": size,
        "modified_at": datetime.fromtimestamp(modified, UTC).isoformat(),
        "digest": r.revision,
        "description": DESCRIPTIONS.get(name, f"{r.family.name} typed-decision model"),
        # TypeSafe clients require a string; fall back to the download date for repos pinned offline.
        "release_date": store.released(r.repo_id) or datetime.fromtimestamp(modified, UTC).date().isoformat(),
        "details": {
            "family": r.family.name,
            "format": "gguf" if r.gguf else "safetensors",
            "quantization_level": quant_of(r.gguf) if r.gguf else None,
        },
        "limits": r.family.limits(r),
    }


def tags() -> list[dict[str, Any]]:
    """One entry per downloaded weight file: a GGUF repo with two quants on disk is two models."""
    out = []
    for repo_id, (size, modified) in sorted(store.downloaded().items()):
        revision = store.pins()[repo_id]
        rev = store.snapshot(repo_id, revision)
        if rev is None:
            continue
        snap = Path(rev.snapshot_path)
        ggufs = sorted(p.name for p in snap.glob("**/*.gguf"))
        names = [f"{repo_id}:{quant_of(g) or g}" for g in ggufs] or [repo_id]
        for name in names:
            try:
                r = store.resolve(name, online=False)
            except (LookupError, ValueError):
                continue
            weight = (snap / r.gguf).stat().st_size if r.gguf else size
            out.append(_describe(canonical(r), r, weight, modified))
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
        r = store.resolve(lookup(req.model), online=False)
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    return {
        "model": canonical(r),
        "repo": r.repo_id,
        "revision": r.revision,
        "file": r.gguf,
        "family": r.family.name,
        "runs_repo_code": r.family.runs_repo_code,
        "trusted": store.is_trusted(r),
        "release_date": store.released(r.repo_id),
        "limits": r.family.limits(r),
        "path": store.local_path(r),
    }


@router.post("/pull")
def api_pull(req: PullRequest) -> Any:
    """Download a model; with stream=true, NDJSON status lines like Ollama's pull."""
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def work() -> None:
        key = None
        try:
            events.put({"status": "pulling manifest"})
            r = store.resolve(lookup(req.model))
            if r.family.runs_repo_code and not store.is_trusted(r):
                events.put(
                    {
                        "error": f"{canonical(r)} runs Python code from its repo; trust is not available over HTTP. "
                        f"Review https://huggingface.co/{r.repo_id}/tree/{r.revision}, then run: "
                        f"ollajev pull {canonical(r)} --trust"
                    }
                )
                return
            key = r.repo_id
            with _pulling_guard:
                busy = key in _pulling
                _pulling.add(key)
            if busy:
                key = None
                events.put({"error": f"{r.repo_id} is already being pulled"})
                return
            events.put({"status": f"downloading {r.repo_id}@{r.revision[:12]}", "digest": r.revision})
            store.download(r)
            prefetch = getattr(r.family, "prefetch", None)
            if prefetch:
                events.put({"status": "downloading base model"})
                prefetch(store.local_path(r))
            events.put({"status": "success", "model": canonical(r)})
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
        r = store.resolve(lookup(req.model), online=False)
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    api.current_manager().unload(canonical(r))
    return {"status": "success", "freed": store.remove(r)}


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
        r = store.resolve(lookup(req.model), online=False)
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    return {"status": "success" if api.current_manager().unload(canonical(r)) else "not loaded"}
