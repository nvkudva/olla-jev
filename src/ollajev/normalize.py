"""Bring every adapter's answers to the one TypeSafe System One answer shape.

Models disagree on extras (x_p_max, decision, level_fit, source) and on what they leave out (some
give no confidence, some key noul probabilities by yes/no). The wire shape is fixed:
  noul    {type, noul, confidence}
  choice  {type, choice, confidence, probabilities}
  score   {type, score, confidence, legend, probabilities}
plus `action` when the model reports one. Confidence always uses TypeSafe's formulas, so it means
the same thing whichever model answered.
"""

from __future__ import annotations

from typing import Any

ND = 4


def _normalised(probabilities: list[float]) -> list[float]:
    total = sum(probabilities)
    return [1.0 / len(probabilities)] * len(probabilities) if total <= 0 else [x / total for x in probabilities]


def choice_confidence(probabilities: list[float]) -> float:
    """(n * p_max - 1) / (n - 1): 0 for a uniform distribution, 1 for all mass on one option."""
    count = len(probabilities)
    if count <= 1:
        return 1.0
    probabilities = _normalised(probabilities)
    return min(1.0, max(0.0, (count * max(probabilities) - 1) / (count - 1)))


def score_confidence(probabilities: list[float]) -> float:
    """1 - expected distance from the most likely level, over the mean distance of levels from the middle."""
    count = len(probabilities)
    if count <= 1:
        return 1.0
    probabilities = _normalised(probabilities)
    most_likely = max(range(count), key=probabilities.__getitem__)
    spread = sum(share * abs(i - most_likely) for i, share in enumerate(probabilities))
    uniform = sum(abs(i - (count - 1) / 2) for i in range(count)) / count
    return min(1.0, max(0.0, 1.0 - spread / uniform))


def answer(question: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    kind = question["type"]
    out: dict[str, Any] = {"type": kind}
    if kind == "noul":
        p_true = float(raw["noul"])
        out["noul"] = round(p_true, ND)
        out["confidence"] = round(choice_confidence([1 - p_true, p_true]), ND)
    elif kind == "choice":
        names = list(question["criteria"])
        probs = raw.get("probabilities") or {}
        probabilities = _normalised([float(probs.get(n, 0.0)) for n in names])
        out["choice"] = (
            raw.get("choice")
            if raw.get("choice") in names
            else names[max(range(len(probabilities)), key=probabilities.__getitem__)]
        )
        out["confidence"] = round(choice_confidence(probabilities), ND)
        out["probabilities"] = {n: round(x, ND) for n, x in zip(names, probabilities, strict=False)}
    elif kind == "score":
        levels = question["criteria"]
        probs = raw.get("probabilities") or {}
        probabilities = _normalised([float(probs.get(str(i), 0.0)) for i in range(len(levels))])
        out["score"] = round(sum(i * x for i, x in enumerate(probabilities)), ND)
        out["confidence"] = round(score_confidence(probabilities), ND)
        out["legend"] = {str(i): level for i, level in enumerate(levels)}
        out["probabilities"] = {str(i): round(x, ND) for i, x in enumerate(probabilities)}
    else:
        raise ValueError(f"unknown question type {kind!r}")
    if "action" in raw:
        out["action"] = raw["action"]
    return out


def answers(questions: dict[str, dict[str, Any]], raw: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = set(questions) - set(raw)
    if missing:
        raise RuntimeError(f"model returned no answer for {sorted(missing)}")
    for qid, reply in raw.items():
        if isinstance(reply, dict) and "error" in reply:
            raise ValueError(f"question {qid!r}: {reply['error']}")
    return {qid: answer(q, raw[qid]) for qid, q in questions.items()}
