"""com-kotobalabs/open-jev-*: imports the `typed_decisions` package bundled in the repo."""

from __future__ import annotations

from typing import Any, cast

from . import import_from, instructions_or_name, text_state
from .base import Loaded

LIMITS = {"max_options": 255, "max_levels": 10, "max_tokens": 512, "languages": "English"}


def _option(name: str, description: Any) -> str:
    return name if description in (None, "") else f"{name}: {text_state(description)}"


def _request(name: str, question: dict[str, Any]) -> tuple[dict[str, Any], list[str] | None]:
    """A Jev question as an open-jev request, with the Jev keys of its options (None for noul)."""
    kind, criteria = question["type"], cast(Any, question.get("criteria"))
    if kind == "noul":
        text = question.get("instructions") or (criteria or {}).get("true") or name.replace("_", " ")
        return {"type": "noul", "instructions": text_state(text)}, None
    instructions = text_state(instructions_or_name(name, question))
    if kind == "choice":
        options = [_option(option, description) for option, description in criteria.items()]
        return {"type": "choice", "instructions": instructions, "options": options}, list(criteria)
    options = [text_state(level) for level in criteria]
    return {"type": "score", "instructions": instructions, "options": options}, [str(i) for i in range(len(criteria))]


def _relabel(output: dict[str, Any], option_keys: list[str] | None) -> dict[str, Any]:
    """open-jev labels probabilities by option text; Jev wants the option keys."""
    if option_keys is None:
        return output
    probabilities = dict(zip(option_keys, output["probabilities"].values(), strict=False))
    answer = {**output, "probabilities": probabilities}
    if "choice" in answer:
        answer["choice"] = max(probabilities, key=probabilities.__getitem__)
    return answer


class _OpenJev:
    name = "open-jev"
    runs_repo_code = True

    def limits(self, resolved) -> dict:
        return LIMITS

    def matches(self, repo_id: str, files: list[str]) -> bool:
        return "open_jev_config.json" in files

    def allow_patterns(self, resolved) -> None:
        return None

    def load(self, path: str, resolved, device: str | None) -> Loaded:
        OpenJev = import_from(path, "typed_decisions.open_jev").OpenJev
        model = OpenJev.from_pretrained(path, device=device)

        def predict(state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
            requests = [_request(name, question) for name, question in questions.items()]
            outputs = model.decide(text_state(state), [request for request, _ in requests])
            answers = {}
            for name, (_, option_keys), output in zip(questions, requests, outputs, strict=False):
                answers[name] = _relabel(output, option_keys)
            return {"answers": answers}

        return Loaded(resolved.name, "open-jev DeBERTa-v3 typed-decision encoder", None, self.limits(resolved), predict)


FAMILY = _OpenJev()
