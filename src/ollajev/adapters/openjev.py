"""com-kotobalabs/open-jev-*: imports the `typed_decisions` package bundled in the repo."""

from __future__ import annotations

from typing import Any, cast

from . import import_from, instructions_or_name, text_state
from .base import Loaded

LIMITS = {"max_options": 255, "max_levels": 10, "max_tokens": 512, "languages": "English"}


def _option(name: str, description: Any) -> str:
    return name if description in (None, "") else f"{name}: {text_state(description)}"


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
            ids, batch, keys = [], [], []
            for qid, q in questions.items():
                kind, crit = q["type"], cast(Any, q.get("criteria"))
                if kind == "noul":
                    text = q.get("instructions") or (crit or {}).get("true") or qid.replace("_", " ")
                    batch.append({"type": "noul", "instructions": text_state(text)})
                    keys.append(None)
                elif kind == "choice":
                    batch.append(
                        {
                            "type": "choice",
                            "instructions": text_state(instructions_or_name(qid, q)),
                            "options": [_option(n, d) for n, d in crit.items()],
                        }
                    )
                    keys.append(list(crit))
                else:
                    batch.append(
                        {
                            "type": "score",
                            "instructions": text_state(instructions_or_name(qid, q)),
                            "options": [text_state(level) for level in crit],
                        }
                    )
                    keys.append([str(i) for i in range(len(crit))])
                ids.append(qid)
            out = model.decide(text_state(state), batch)
            answers = {}
            for qid, option_keys, answer in zip(ids, keys, out, strict=False):
                if option_keys is not None:
                    probabilities = dict(zip(option_keys, answer["probabilities"].values(), strict=False))
                    answer = {**answer, "probabilities": probabilities}
                    if "choice" in answer:
                        answer["choice"] = max(probabilities, key=probabilities.__getitem__)
                answers[qid] = answer
            return {"answers": answers}

        return Loaded(resolved.name, "open-jev DeBERTa-v3 typed-decision encoder", None, self.limits(resolved), predict)


FAMILY = _OpenJev()
