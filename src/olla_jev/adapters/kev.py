"""jaredpalmer/kev-*: LoRA adapter + pointer head on a Qwen base, scored by kev's own loader.

The repos ship no code; kev's loader is vendored from GitHub at a pinned commit (olla_jev/_vendor/kev).
The base model is fetched at the revision head.pt names.
"""

from __future__ import annotations

from typing import Any

from .base import Loaded

LIMITS = {"max_options": 255, "max_levels": 10, "max_tokens": 8192}


class _Kev:
    name = "kev"
    runs_repo_code = False

    def limits(self, r) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return "head.pt" in files and "adapter_config.json" in files

    def allow_patterns(self, r) -> list[str]:
        return ["*.json", "*.safetensors", "head.pt", "*.txt", "*.jinja"]

    def prefetch(self, path: str) -> None:
        """Download the base model head.pt names, so `pull` leaves nothing to fetch at load time."""
        import torch
        from huggingface_hub import snapshot_download

        meta = torch.load(f"{path}/head.pt", map_location="cpu", weights_only=True)
        snapshot_download(
            meta["base"],
            revision=meta.get("base_revision"),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "merges.txt", "vocab.json"],
        )

    def load(self, path: str, r, device: str | None) -> Loaded:
        import torch

        from .._vendor.kev.api import SystemOneRequest, to_answers, to_record
        from .._vendor.kev.checkpoint import Checkpoint, LoadOptions
        from .._vendor.kev.model import admit

        device = device or "cpu"
        # kev.serve's own defaults: bf16 off the CPU, sdpa attention on Apple GPUs.
        opts = LoadOptions(dtype=None if device == "cpu" else torch.bfloat16, attn="sdpa" if device == "mps" else None)
        ck = Checkpoint(path)
        tok, model = ck.load(device, opts)

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            rec, meta = to_record(SystemOneRequest.model_validate({"state": state, "questions": questions}))
            enc = admit(model, tok, rec)
            with torch.inference_mode():
                probs = model.probs(enc)
            return {
                "answers": to_answers([p.tolist() for p in probs], meta),
                "usage": {"input_tokens": len(enc["ids"]), "output_tokens": 0},
            }

        return Loaded(r.name, f"Kev pointer head on {ck.meta.base}", None, self.limits(r), predict)


FAMILY = _Kev()
