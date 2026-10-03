"""convaiinnovations/laya*: the `laya` PyPI package. No repo code is imported."""

from __future__ import annotations

from typing import Any

from . import instructions_or_name
from .base import Loaded

# Context lengths from the model cards; laya does not expose them.
CONTEXT = {"convaiinnovations/laya": 512, "convaiinnovations/laya-multilingual": 1024}
ALLOW = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]


class _Laya:
    name = "laya"
    runs_repo_code = False

    def limits(self, r) -> dict:
        return {"max_tokens": CONTEXT.get(r.repo_id, 1024)}

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return "rl_agent_config.json" in files and "model.safetensors" in files

    def allow_patterns(self, r) -> list[str]:
        return ALLOW

    def load(self, path: str, r, device: str | None) -> Loaded:
        import laya
        from transformers.initialization import no_init_weights

        # from_config randomly initialises the encoder, then a strict load_state_dict overwrites every
        # tensor of it; skipping the init saves ~24 s and is safe because that load is strict.
        with no_init_weights():
            agent = laya.load(path, device=device)

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            qs = {}
            for qid, q in questions.items():
                out = {"type": q["type"], "instructions": instructions_or_name(qid, q)}
                if q.get("criteria") is not None:
                    out["criteria"] = q["criteria"]
                qs[qid] = out
            return agent.system_one(state, qs)

        return Loaded(r.name, "Laya typed-decision encoder", None, self.limits(r), predict)


FAMILY = _Laya()
