"""Ollama-style model management: /api/tags, /api/ps, /api/pull, /api/show, /api/delete, /api/copy, /api/stop."""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import config, store
from .catalog import CATALOG
from .manager import canonical, lookup
from .names import quant_of

router = APIRouter(prefix="/api")
DESCRIPTIONS = {e.name: e.description for e in CATALOG}


class ModelRef(BaseModel):
    model: str


class PullRequest(BaseModel):
    model: str
    stream: bool = True
    trust: bool = False


class CopyRequest(BaseModel):
    source: str
    destination: str


def _snapshot_dir(repo_id: str, revision: str) -> Path | None:
    for repo in store.scan_cache_dir(config.models_dir()).repos:
        if repo.repo_id == repo_id:
            for rev in repo.revisions:
                if rev.commit_hash == revision:
                    return Path(rev.snapshot_path)
    return None


def _describe(name: str, r: store.Resolved, size: int, modified: float) -> dict[str, Any]:
    return {
        "name": name,
        "model": name,
        "size": size,
        "modified_at": datetime.fromtimestamp(modified, timezone.utc).isoformat(),
        "digest": r.revision,
        "description": DESCRIPTIONS.get(name, f"{r.family.name} typed-decision model"),
        # TypeSafe clients require a string; fall back to the download date for repos pinned offline.
        "release_date": store.released(r.repo_id) or datetime.fromtimestamp(modified, timezone.utc).date().isoformat(),
        "details": {"family": r.family.name, "format": "gguf" if r.gguf else "safetensors",
                    "quantization_level": quant_of(r.gguf) if r.gguf else None},
        "limits": r.family.limits(r),
    }


def tags() -> list[dict[str, Any]]:
    """One entry per downloaded weight file: a GGUF repo with two quants on disk is two models."""
    out = []
    for repo_id, (size, modified) in sorted(store.downloaded().items()):
        revision = store.pins()[repo_id]
        snap = _snapshot_dir(repo_id, revision)
        if snap is None:
            continue
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

    now = __import__("time").monotonic()
    models = []
    for slot in api.manager.loaded():
        expires = None if slot.expires == float("inf") else datetime.fromtimestamp(
            datetime.now().timestamp() + (slot.expires - now), timezone.utc).isoformat()
        models.append({"name": slot.name, "model": slot.name, "device": slot.device, "expires_at": expires,
                       "details": {"family": slot.resolved.family.name}})
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
        try:
            events.put({"status": "pulling manifest"})
            r = store.resolve(lookup(req.model))
            if r.family.runs_repo_code and not store.is_trusted(r):
                if not req.trust:
                    events.put({"error": f"{canonical(r)} runs Python code from its repo; pull again with trust=true "
                                         f"after reviewing https://huggingface.co/{r.repo_id}/tree/{r.revision}"})
                    return
                store.trust(r)
            events.put({"status": f"downloading {r.repo_id}@{r.revision[:12]}", "digest": r.revision})
            store.download(r)
            prefetch = getattr(r.family, "prefetch", None)
            if prefetch:
                events.put({"status": "downloading base model"})
                prefetch(store.local_path(r))
            events.put({"status": "success", "model": canonical(r)})
        except Exception as exc:  # reported to the client as an error line
            events.put({"error": str(exc)})
        finally:
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

    aliases = config.load().get("aliases", {})
    if req.model in aliases:
        aliases.pop(req.model)
        config.update(aliases=aliases)
        return {"status": "success"}
    try:
        r = store.resolve(lookup(req.model), online=False)
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    api.manager.unload(canonical(r))
    path = store.local_path(r)
    others = [p for p in Path(path).glob("**/*.gguf") if p.name != Path(r.gguf).name] if r.gguf and path else []
    if others:  # other quants of this repo stay; remove only this file and its blob
        link = Path(path) / r.gguf
        blob = link.resolve()
        freed = blob.stat().st_size
        link.unlink()
        blob.unlink()
        return {"status": "success", "freed": freed}
    return {"status": "success", "freed": store.delete(r.repo_id)}


@router.post("/copy")
def api_copy(req: CopyRequest) -> Any:
    aliases = config.load().get("aliases", {})
    aliases[req.destination] = lookup(req.source)
    config.update(aliases=aliases)
    return {"status": "success"}


@router.post("/stop")
def api_stop(req: ModelRef) -> Any:
    from . import api

    try:
        r = store.resolve(lookup(req.model), online=False)
    except LookupError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    return {"status": "success" if api.manager.unload(canonical(r)) else "not loaded"}
