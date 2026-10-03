"""jaredpalmer/kev-*: LoRA adapter + pointer head on a Qwen base, scored by kev's own loader.

The repos ship no code; kev's loader is vendored from GitHub at a pinned commit (ollajev/_vendor/kev).
The base model is fetched at the revision head.pt names.
"""

from __future__ import annotations

import os
from typing import Any

from .base import Loaded

LIMITS = {"max_options": 255, "max_levels": 10, "max_tokens": 8192}


class _Kev:
    name = "kev"
    runs_repo_code = False

    def limits(self, resolved) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return "head.pt" in files and "adapter_config.json" in files

    def allow_patterns(self, resolved) -> list[str]:
        return ["*.json", "*.safetensors", "head.pt", "*.txt", "*.jinja"]

    def prefetch(self, path: str, tqdm_class=None) -> None:
        """Download the base model head.pt names, so `pull` leaves nothing to fetch at load time. It goes to the
        default Hugging Face cache, where the vendored loader looks for it."""
        import torch
        from huggingface_hub import snapshot_download

        meta = torch.load(f"{path}/head.pt", map_location="cpu", weights_only=True)
        snapshot_download(
            meta["base"],
            revision=meta.get("base_revision"),
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "merges.txt", "vocab.json"],
            tqdm_class=tqdm_class,
        )

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        import torch

        from .._vendor.kev.api import SystemOneRequest, to_answers, to_record
        from .._vendor.kev.checkpoint import Checkpoint, LoadOptions
        from .._vendor.kev.model import admit

        device = device or "cpu"
        # kev.serve's own defaults: bf16 off the CPU, sdpa attention on Apple GPUs.
        dtype = None if device == "cpu" else torch.bfloat16
        attention = "sdpa" if device == "mps" else None
        opts = LoadOptions(dtype=dtype, attn=attention)
        ck = Checkpoint(path)
        _use_cached_base(ck.meta)
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

        return Loaded(resolved.name, f"Kev pointer head on {ck.meta.base}", None, self.limits(resolved), predict)


def _use_cached_base(meta: Any) -> None:
    """Point the checkpoint at the base model's downloaded folder instead of its repo name. By name, each
    from_pretrained call (tokenizer, config, weights) first asks Hugging Face for the latest files: kev-0.5b took
    20-30 s to load that way and 2.6 s from the folder. A base that is not downloaded keeps its name and downloads."""
    # Not snapshot_download(local_files_only=True): it refuses a snapshot missing README.md and the like, which
    # prefetch skips on purpose.
    from huggingface_hub import try_to_load_from_cache

    config_file = try_to_load_from_cache(meta.base, "config.json", revision=meta.base_revision)
    if not isinstance(config_file, str):
        return
    meta.base = os.path.dirname(config_file)
    meta.base_revision = None


FAMILY = _Kev()
