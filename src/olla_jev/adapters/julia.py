"""SupersonicLabs/Julia-*: imports the `julia` package bundled in the repo."""

from __future__ import annotations

from typing import Any

from . import import_from, instructions_or_name, text_state
from .base import Loaded

LIMITS = {"max_options": 20, "max_levels": 20, "max_tokens": 8192, "languages": "Multilingual"}


def _label(name: str, description: Any) -> str:
    return name if description in (None, "") else f"{name}: {text_state(description)}"


class _Julia:
    name = "julia"
    runs_repo_code = True

    def limits(self, r) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return "julia_config.json" in files

    def allow_patterns(self, r) -> list[str]:
        return ["*.json", "*.safetensors", "julia/*.py", "julia/router/*.py", "encoder/*", "tokenizer/*"]

    def load(self, path: str, r, device: str | None) -> Loaded:
        load_model = import_from(path, "julia").load_model
        # The fast path rewrites ModernBERT's forward around a private method transformers 5.1 removed;
        # the stock forward gives the same outputs.
        import_from(path, "julia.router.encoder").specialize_decision_encoder = lambda model: False
        # Strict encoding rejects an over-long request instead of silently truncating the state.
        # Its runtime moves batches only for CUDA; on MPS inputs stay on the CPU. 144M parameters run fast on CPU.
        device = "cuda" if device == "cuda" else "cpu"
        engine = load_model(path, device=device, max_length=LIMITS["max_tokens"], strict_encoding=True)

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            qs = {}
            for qid, q in questions.items():
                crit: Any = q.get("criteria")
                out = {"type": q["type"], "instructions": text_state(instructions_or_name(qid, q))}
                if q["type"] == "choice":
                    out["criteria"] = {n: _label(n, d) for n, d in crit.items()}
                elif q["type"] == "score":
                    out["criteria"] = [text_state(level) for level in crit]
                elif crit:
                    out["criteria"] = {k: _label(k, crit.get(k)) for k in ("false", "true")}
                qs[qid] = out
            return engine.predict(state=state, questions=qs)

        return Loaded(r.name, "Julia mmBERT-small typed-decision encoder", None, self.limits(r), predict, device=device)


FAMILY = _Julia()
