"""Jev-compatible (TypeSafe System One) HTTP API, plus the demo page and the Ollama-style admin API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import admin, normalize, presets
from .manager import Manager, NotDownloaded, NotTrusted, default_model

JSONContent = str | dict[str, Any] | list[Any]



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
    model: str = "jev-latest"  # manager.DEFAULT_ALIASES: means the default model
    questions: dict[str, Question] = Field(min_length=1)


manager: Manager | None = None

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global manager
    manager = Manager()
    if preload:
        # Load before uvicorn accepts connections, so a reply from any route means ready to decide.
        manager.get(preload, keep_alive=-1 if pin_preload else None)
    yield
    manager.unload_all()


preload: str | None = None  # set by `serve` before startup
pin_preload = False

app = FastAPI(title="olla-jev", version=version("olla-jev"), lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _invalid(loc: list[str | int], msg: str, kind: str = "value_error", status: int = 422) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": [{"loc": loc, "msg": msg, "type": kind}]})


def _wire(q: Question) -> dict[str, Any]:
    out = q.model_dump(exclude_none=True)
    if isinstance(q, ChoiceQuestion):
        out["criteria"] = q.criteria  # keep None descriptions; exclude_none would drop those options
    return out


def _model_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, NotDownloaded):
        return _invalid(["body", "model"], str(exc), "model_not_found", 404)
    if isinstance(exc, NotTrusted):
        return _invalid(["body", "model"], str(exc), "model_not_trusted", 403)
    return _invalid(["body", "model"], str(exc))


@app.get("/")
async def health() -> dict[str, Any]:
    return {"status": "ok", "default_model": default_model(), "loaded": [s.name for s in manager.loaded()], "ui": "/demo"}


@app.get("/demo")
async def demo() -> FileResponse:
    return FileResponse(STATIC_DIR / "demo.html", media_type="text/html")


@app.get("/ui/presets")
async def ui_presets() -> dict[str, Any]:
    return presets.examples()


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    """Every downloaded model, default first. TypeSafe clients read name, description and release_date."""
    default = default_model()
    out = []
    for m in admin.tags():
        out.append({"name": m["name"], "description": m["description"], "release_date": m["release_date"],
                    "default": m["name"] == default, "limits": m["limits"]})
    out.sort(key=lambda m: not m["default"])
    return {"models": out}


@app.post("/v1/systemone")
def system_one(req: Annotated[SystemOneRequest, Body()]) -> Any:
    questions = {name: _wire(q) for name, q in req.questions.items()}
    try:
        slot, result = manager.run(req.model, req.state, questions)
        answers = normalize.answers(questions, result["answers"])
    except (NotDownloaded, NotTrusted) as exc:
        return _model_error(exc)
    except ValueError as exc:
        return _invalid(["body", "questions"], str(exc))
    usage = result.get("usage") or {}
    return {
        "model": slot.name,
        "answers": answers,
        "usage": {"input_tokens": int(usage.get("input_tokens", 0)), "output_tokens": int(usage.get("output_tokens", 0))},
    }


app.include_router(admin.router)


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
