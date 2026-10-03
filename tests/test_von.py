"""The von adapter's repo detection and answer shaping, without loading any weights."""

from __future__ import annotations

from von.types import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage

from ollajev import normalize
from ollajev.adapters import detect, von

FILES = [
    "config.json",
    "model.safetensors",
    "option_marker.pt",
    "calibration.json",
    "marker_calibration.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def test_von_repos_are_detected_and_others_are_not():
    assert detect("wfzyx/von", FILES).name == "von"
    assert not von.FAMILY.matches("x/bert", ["config.json", "model.safetensors", "tokenizer.json"])
    assert not von.FAMILY.matches("x/laya", ["rl_agent_config.json", "model.safetensors"])


def test_questions_get_instructions_and_keep_criteria():
    qs = von.questions_for(
        {
            "is_refund": {"type": "noul"},
            "route": {"type": "choice", "instructions": "Team?", "criteria": {"a": None, "b": "B"}},
        }
    )
    assert qs["is_refund"] == {"type": "noul", "instructions": "is refund"}
    assert qs["route"]["criteria"] == {"a": None, "b": "B"}


def test_sdk_answers_normalise_to_the_wire_shape():
    response = SystemOneResponse(
        model="von-1.3.0",
        answers={
            "n": NoulAnswer(noul=0.83, noul_raw=0.8),
            "c": ChoiceAnswer(choice="a", probabilities={"a": 0.7, "b": 0.3}, confidence=0.4),
            "s": ScoreAnswer(
                score=0.9, confidence=0.6, legend={"0": "lo", "1": "hi"}, probabilities={"0": 0.1, "1": 0.9}
            ),
        },
        usage=Usage(input_tokens=42, output_tokens=3),
    )
    out = von.answers_from(response)
    assert out["usage"] == {"input_tokens": 42, "output_tokens": 0}
    questions = {
        "n": {"type": "noul"},
        "c": {"type": "choice", "criteria": {"a": None, "b": None}},
        "s": {"type": "score", "criteria": ["lo", "hi"]},
    }
    wire = normalize.answers(questions, out["answers"])
    assert wire["n"]["noul"] == 0.83
    assert wire["c"]["choice"] == "a"
    assert wire["c"]["probabilities"] == {"a": 0.7, "b": 0.3}
    assert wire["s"]["score"] == 0.9
