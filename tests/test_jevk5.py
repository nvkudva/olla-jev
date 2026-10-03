"""JevK5 adapter: matching, prompt building, temperatures and answer shaping, all offline."""

from __future__ import annotations

import json

import pytest

from ollajev import normalize, store
from ollajev._vendor.jevk5 import prompt
from ollajev.adapters import detect, jevk5

SAFETENSORS = ["config.json", "model.safetensors", "jevk5_config.json", "chat_template.jinja", "tokenizer.json"]


def test_matches_only_jevk5_repos():
    assert detect("alibiserikbay/JevK5-2B", SAFETENSORS).name == "jevk5"
    assert not jevk5.FAMILY.matches(
        "alibiserikbay/JevK5-Lite", ["config.json", "lite_config.json", "model.safetensors"]
    )
    assert not jevk5.FAMILY.matches("Mapika/decider-2b", ["decider_config.json", "config.json", "model.safetensors"])
    assert not jevk5.FAMILY.matches("u/x", ["jevk5_config.json", "README.md"])  # no weights


def test_the_gguf_repo_runs_on_its_base(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.setenv("OLLAJEV_MODELS", str(tmp_path / "models"))
    ggufs = ["jevk5-2b-v0.2-Q8_0.gguf", "jevk5-4b-v0.3-Q8_0.gguf", "jevk5-4b-v0.3-Q4_K_M.gguf", "README.md"]
    repos = {
        "alibiserikbay/JevK5-GGUF": ("c1", ggufs, ["alibiserikbay/JevK5"]),
        "alibiserikbay/JevK5": ("b1", SAFETENSORS, []),
    }
    monkeypatch.setattr(store, "_remote_files", lambda repo, rev: (repos[repo][0], None, *repos[repo][1:]))
    r = store.resolve("alibiserikbay/JevK5-GGUF:jevk5-2b-v0.2-Q8_0.gguf")
    assert (r.family.name, r.weights, r.allow) == ("jevk5", "jevk5-2b-v0.2-Q8_0.gguf", ["jevk5-2b-v0.2-Q8_0.gguf"])
    assert r.base is not None and r.base.repo_id == "alibiserikbay/JevK5" and r.base.allow == ["jevk5_config.json"]


def test_temperatures_come_from_the_gguf_file_else_the_config(tmp_path):
    (tmp_path / "jevk5_config.json").write_text(json.dumps({"temperature": 1.22, "knockout_temperature": 0.93}))
    assert jevk5.temperatures(str(tmp_path)) == (1.22, 0.93)
    assert jevk5.temperatures(str(tmp_path), "jevk5-2b-v0.2-Q8_0.gguf") == (1.42, 0.77)
    assert jevk5.temperatures(str(tmp_path), "jevk5-9b-v0.3.3-Q5_K_M.gguf") == (1.316, 1.05)
    assert jevk5.temperatures(str(tmp_path), "jevk5-4b-v0.4-Q8_0.gguf") == (1.22, 0.93)  # unknown file: the base's
    (tmp_path / "jevk5_config.json").write_text(json.dumps({"temperature": 1.42}))
    assert jevk5.temperatures(str(tmp_path)) == (1.42, None)


def test_prompt_follows_the_runtime_layout():
    text = prompt.prompt_text({"ticket": "refund"}, "Which team?", ["billing: Payments", "tech: tech"])
    assert text.startswith("<|im_start|>system\nApply the supplied criterion")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    user = text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
    assert json.loads(user) == {
        "evidence": {"ticket": "refund"},
        "criterion": "Which team?",
        "options": [{"letter": "A", "description": "billing: Payments"}, {"letter": "B", "description": "tech: tech"}],
    }


def test_wire_questions_become_lettered_options():
    noul = jevk5.question("is_refund", {"type": "noul"})
    assert noul["instructions"] == "is refund"
    assert prompt.decision_options(noul) == [
        ("true", "true: The proposition is true."),
        ("false", "false: The proposition is false."),
    ]
    choice = jevk5.question("q", {"type": "choice", "instructions": "x", "criteria": ["a", "b"]})
    assert prompt.decision_options(choice) == [("a", "a: a"), ("b", "b: b")]
    score = jevk5.question("q", {"type": "score", "instructions": "x", "criteria": ["low", "high"]})
    assert prompt.decision_options(score) == [("0", "0: low"), ("1", "1: high")]


def test_letter_logits_become_calibrated_answers():
    questions = {
        "yes": {"type": "noul", "instructions": "Refund?"},
        "team": {"type": "choice", "instructions": "Team?", "criteria": {"billing": "Pay", "tech": None, "sales": ""}},
        "level": {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]},
    }
    prompts = []

    def encode(text):
        prompts.append(text)
        return list(range(10))

    def letter_logits(ids, count):
        return [2.0, 0.0, -2.0][:count]

    out = jevk5.system_one(encode, letter_logits, 2.0, None, "billed twice", questions)
    assert out["usage"] == {"input_tokens": 30, "output_tokens": 0}
    answers = normalize.answers(questions, out["answers"])
    assert answers["yes"]["noul"] == pytest.approx(0.7311, abs=1e-4)  # softmax([2, 0] / 2): true is letter A
    assert answers["team"]["choice"] == "billing"
    assert sum(answers["team"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert answers["level"]["score"] < 1.0
    assert '"description": "tech: tech"' in prompts[1] and '"description": "sales: sales"' in prompts[1]


def test_overlong_prompts_are_refused():
    with pytest.raises(ValueError, match="at most"):
        jevk5.system_one(
            lambda text: [0] * (jevk5.MAX_TOKENS + 1),
            lambda ids, n: [0.0] * n,
            1.0,
            None,
            "x",
            {"q": {"type": "noul", "instructions": "y"}},
        )


def test_more_than_16_options_take_several_passes():
    passes = []

    def letter_logits(ids, count):
        passes.append(count)
        return [float(-i) for i in range(count)]

    names = [f"o{i}" for i in range(40)]
    q = {"q": {"type": "choice", "instructions": "pick", "criteria": names}}
    out = jevk5.system_one(lambda text: [1], letter_logits, 1.0, 0.93, "s", q)
    assert len(passes) == 4 and max(passes) <= 16
    assert sum(out["answers"]["q"]["probabilities"].values()) == pytest.approx(1.0)
