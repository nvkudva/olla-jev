"""Jev-compatible (TypeSafe System One) HTTP API, served by whichever adapter is loaded."""

from __future__ import annotations

import threading
from importlib.metadata import version
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .adapters.base import Adapter

JSONContent = str | dict[str, Any] | list[Any]

# typesafe-sdk's DEFAULT_MODEL. A client that never passes `model=` sends it, so it resolves to
# whichever model this process serves.
DEFAULT_ALIAS = "jev-latest"


class NoulCriteria(BaseModel):
    true: JSONContent | None = None
    false: JSONContent | None = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent | None = None
    criteria: dict[str, JSONContent | None] = Field(min_length=2)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: JSONContent | None = None
    criteria: list[JSONContent] = Field(min_length=2)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = DEFAULT_ALIAS
    questions: dict[str, Question] = Field(min_length=1)


# One forward pass at a time. On MPS concurrent forwards abort the process with a Metal
# command-buffer assertion, so this is a correctness requirement, not a throttle.
_lock = threading.Lock()
_adapter: Adapter | None = None

app = FastAPI(title="free-jev-server", version=version("free-jev-server"))


def use_adapter(adapter: Adapter) -> None:
    """Pick the model this process serves. Call before startup."""
    global _adapter
    _adapter = adapter


def _invalid(loc: list[str | int], msg: str, kind: str = "value_error") -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": [{"loc": loc, "msg": msg, "type": kind}]})


def _wire(q: Question) -> dict[str, Any]:
    out = q.model_dump(exclude_none=True)
    if isinstance(q, ChoiceQuestion):
        out["criteria"] = q.criteria  # keep None descriptions; exclude_none would drop those options
    return out


@app.get("/")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model": _adapter.name if _adapter else None}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    if _adapter is None:
        return {"models": []}
    return {"models": [{"name": _adapter.name, "description": _adapter.description, "release_date": _adapter.released}]}


@app.post("/v1/systemone")
def system_one(req: Annotated[SystemOneRequest, Body()]) -> Any:
    if _adapter is None:
        return JSONResponse(status_code=503, content={"detail": [{"loc": [], "msg": "no model loaded", "type": "unavailable"}]})
    if req.model not in (_adapter.name, DEFAULT_ALIAS):
        return _invalid(["body", "model"], f"this server serves {_adapter.name!r}, not {req.model!r}")
    questions = {name: _wire(q) for name, q in req.questions.items()}
    try:
        with _lock:
            result = _adapter.system_one(req.state, questions)
    except ValueError as exc:
        return _invalid(["body", "questions"], str(exc))
    return {"model": _adapter.name, "answers": result["answers"], "usage": result.get("usage", {"input_tokens": 0, "output_tokens": 0})}


@app.exception_handler(StarletteHTTPException)
def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Starlette answers 404/405 with a bare string; give them the same shape as every other error."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": [{"loc": [], "msg": exc.detail, "type": "http_error"}]},
        headers=exc.headers,  # 405 carries Allow
    )


@app.exception_handler(Exception)
def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500, content={"detail": [{"loc": ["body"], "msg": "internal error", "type": "internal_error"}]}
    )
