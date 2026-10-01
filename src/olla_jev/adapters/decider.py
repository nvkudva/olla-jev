"""Mapika/decider-*: the `decider-ai` PyPI package, on PyTorch or on llama.cpp for GGUF repos.

The repos bundle a copy of `decider/`, but the pinned PyPI release is the same code and also carries
the GGUF engine, so no repo code is imported.
"""

from __future__ import annotations

from typing import Any

from .base import Loaded

LIMITS = {"max_options": 255, "max_levels": 10, "max_tokens": 32768}


class _Decider:
    name = "decider"
    runs_repo_code = False

    def limits(self, r) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return "decider_config.json" in files

    def allow_patterns(self, r) -> list[str]:
        meta = ["*.json", "*.jinja", "*.txt", "tokenizer*"]
        return [r.gguf, *meta] if r.gguf else ["*.safetensors", *meta]

    def load(self, path: str, r, device: str | None) -> Loaded:
        from decider.infer import Decider

        if r.gguf:
            d = Decider(path, gguf_file=r.gguf)
            backend = "llama.cpp"
        else:
            d = Decider(path, device=device)
            backend = f"PyTorch {device}"

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            return d.system_one(state, questions)

        return Loaded(
            r.name,
            f"decider typed-decision model ({backend})",
            None,
            self.limits(r),
            predict,
            device="llama.cpp" if r.gguf else device,
        )


FAMILY = _Decider()
