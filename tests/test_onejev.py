"""OneJev's prompt, answer shaping and repo matching, without loading any weights."""

from __future__ import annotations

import pytest

from ollajev import normalize, store
from ollajev.adapters import detect, onejev

FILES = ["config.json", "model.safetensors", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]


def test_choice_turn_matches_qev_labels_v2():
    q = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "refunds", "sales": None}}
    labels, suffix = onejev.question(q)
    assert labels == ["billing", "sales"]
    assert suffix == "Question: Which team?\n\nOptions:\nA. billing: refunds\nB. sales\n\nAnswer with one letter: A, B."


def test_noul_and_score_turns_use_the_qev_defaults():
    labels, suffix = onejev.question({"type": "noul"})
    assert labels == ["yes", "no"]
    assert suffix.startswith(f"Question: {onejev.NOUL_INSTRUCTIONS}\n\nOptions:\nA. yes: {onejev.NOUL_TRUE}\n")
    labels, suffix = onejev.question({"type": "score", "instructions": "How upset?", "criteria": ["calm", {"x": 1}]})
    assert labels == ["0", "1"]
    assert suffix.startswith("Rate the state: How upset?\n\nOptions:\nA. level 0: calm\nB. level 1: {\n")


def test_many_options_get_two_letter_labels():
    labels, suffix = onejev.question({"type": "choice", "criteria": [f"o{i}" for i in range(30)]})
    assert len(labels) == 30 and suffix.endswith("Answer with one label: " + ", ".join(onejev.SLOTS[:30]) + ".")
    assert len(onejev.SLOTS) == 255 and len(set(onejev.SLOTS)) == 255
    with pytest.raises(ValueError):
        onejev.question({"type": "choice", "criteria": [f"o{i}" for i in range(256)]})


def test_state_goes_first_so_questions_share_a_prefix():
    msgs = onejev.messages({"a": 1}, "Question: q")
    assert msgs[0] == {"role": "system", "content": onejev.SYSTEM}
    assert msgs[1]["content"] == '<state>\n{\n  "a": 1\n}\n</state>\n\nQuestion: q'


def test_slot_logits_become_normalised_answers():
    noul = onejev.shape({"type": "noul"}, ["yes", "no"], [2.0, 0.0])
    assert noul["noul"] == pytest.approx(0.8808, abs=1e-4)
    q = {"type": "choice", "criteria": {"a": "", "b": "", "c": ""}}
    raw = onejev.shape(q, ["a", "b", "c"], [0.0, 3.0, 1.0])
    assert raw["choice"] == "b" and sum(raw["probabilities"].values()) == pytest.approx(1.0)
    q = {"type": "score", "criteria": ["low", "high"]}
    out = normalize.answer(q, onejev.shape(q, ["0", "1"], [0.0, 0.0]))
    assert out["score"] == 0.5 and out["probabilities"] == {"0": 0.5, "1": 0.5}


def test_matches_only_onejev_repos():
    for repo in ["OmniJev/OneJev-0.8B", "OmniJev/OneJev-4B", "OmniJev/OneJev-27B"]:
        assert detect(repo, FILES).name == "onejev"
    sharded = [f for f in FILES if f != "model.safetensors"] + ["model.safetensors.index.json"]
    assert onejev.FAMILY.matches("OmniJev/OneJev-27B", sharded)
    assert not onejev.FAMILY.matches("OmniJev/OneJev-27B-FP8", sharded)  # compressed-tensors FP8
    assert not onejev.FAMILY.matches("Qwen/Qwen3.5-0.8B", FILES)
    assert not onejev.FAMILY.matches("bartowski/OmniJev_OneJev-0.8B-GGUF", ["OmniJev_OneJev-0.8B-Q8_0.gguf"])
    assert not onejev.FAMILY.matches("OmniJev/OneJev-0.8B", ["README.md"])
    assert detect("Mapika/decider-0.8b", ["decider_config.json", *FILES]).name == "decider"


def test_gguf_copies_run_on_the_base_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.setenv("OLLAJEV_MODELS", str(tmp_path / "models"))
    repos = {
        "mradermacher/OneJev-0.8B-GGUF": ("c1", ["OneJev-0.8B.Q8_0.gguf", "OneJev-0.8B.mmproj-f16.gguf"]),
        "OmniJev/OneJev-0.8B": ("b1", FILES),
    }
    bases = {"mradermacher/OneJev-0.8B-GGUF": ["OmniJev/OneJev-0.8B"]}
    monkeypatch.setattr(
        store, "_remote_files", lambda repo, rev: (repos[repo][0], None, repos[repo][1], bases.get(repo, []))
    )
    r = store.resolve("mradermacher/OneJev-0.8B-GGUF:Q8_0")
    assert (r.family.name, r.weights) == ("onejev", "OneJev-0.8B.Q8_0.gguf")
    assert r.base is not None and r.base.repo_id == "OmniJev/OneJev-0.8B" and r.base.allow == onejev.META
