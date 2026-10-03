"""Cloudflare/clef and clef-flash: a Qwen3.5 backbone with a joint schema head, scored in one forward pass.

The repos ship `joint_schema_model.py`; an unchanged copy is vendored (ollajev/_vendor/clef), so no repo
code is imported. Text states only; the vision tower loads with the checkpoint but stays unused.
"""

from __future__ import annotations

from typing import Any

from .base import Loaded

HEAD = ("joint_head_config.json", "joint_head.safetensors")
LIMITS = {"max_options": 255, "max_levels": 255, "max_tokens": 16384, "languages": "Multilingual"}


def record(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The model's input record. Wire questions keep their shape; a noul keeps the model's default wording for
    whichever of true and false the request does not describe."""
    qs = {}
    for qid, q in questions.items():
        out = {k: q[k] for k in ("type", "instructions", "criteria") if q.get(k) is not None}
        if q["type"] == "noul":
            crit = q.get("criteria")
            crit = crit if isinstance(crit, dict) else {}
            out["criteria"] = {k: v for k, v in crit.items() if k in ("true", "false") and v is not None}
        qs[qid] = out
    return {"state": state, "questions": qs}


def answer(kind: str, probabilities: dict[str, float]) -> dict[str, Any]:
    """One question's option probabilities as an adapter answer; normalize adds confidence and the legend."""
    if kind == "noul":
        return {"noul": probabilities["true"]}
    if kind == "choice":
        return {"choice": max(probabilities, key=probabilities.__getitem__), "probabilities": probabilities}
    return {"probabilities": probabilities}


class _Clef:
    name = "clef"
    runs_repo_code = False

    def limits(self, r) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        """Cloudflare's layout in bf16. MLX, vLLM, EXL3, OpenVINO and int8 copies keep the joint head but ship
        their own runtime script, and llm-compressor (FP8/NVFP4) copies a recipe.yaml; transformers runs none."""
        official = all(f in files for f in (*HEAD, "joint_schema_model.py", "model.safetensors.index.json"))
        scripts = [f for f in files if f.endswith(".py") and f != "joint_schema_model.py"]
        return official and not scripts and "recipe.yaml" not in files

    def allow_patterns(self, r) -> list[str]:
        return ["*.json", "*.safetensors", "*.jinja", "tokenizer*"]

    def load(self, path: str, r, device: str | None) -> Loaded:
        import torch

        from .._vendor.clef import joint_schema_model as clef

        device = device or "cpu"
        model, processor = clef.load_release_model(
            path, device=device, dtype=torch.float32 if device == "cpu" else torch.bfloat16
        )
        tok = processor.tokenizer

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            enc = clef.encode_record(tok, record(state, questions), max_length=1 << 30)
            if len(enc.input_ids) > LIMITS["max_tokens"]:
                raise ValueError(f"request is {len(enc.input_ids)} tokens; this model takes {LIMITS['max_tokens']}")
            batch = clef.collate_records([enc], tok.pad_token_id, torch.device(device))
            with torch.inference_mode():
                logits = model(batch)[0]
            answers = {
                q.question_id: answer(
                    questions[q.question_id]["type"],
                    dict(zip(q.option_ids, x.float().softmax(-1).tolist(), strict=True)),
                )
                for q, x in zip(enc.questions, logits, strict=True)
            }
            return {"answers": answers, "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}}

        return Loaded(
            r.name, f"Clef joint schema model (PyTorch {device})", None, self.limits(r), predict, device=device
        )


FAMILY = _Clef()
