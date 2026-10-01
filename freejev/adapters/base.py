"""The contract every model family's adapter implements."""

from __future__ import annotations

from typing import Any, Protocol


class Adapter(Protocol):
    name: str
    description: str
    released: str | None

    def system_one(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Answer every question about `state`.

        `questions` is the request's map in wire form: {id: {"type", "instructions"?, "criteria"?}}.
        Returns {"answers": {id: answer}, "usage"?: {...}}. Raise ValueError for a request this
        model cannot take (too many options, too long); it becomes a 422.
        """
        ...
