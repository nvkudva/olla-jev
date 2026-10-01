"""The wire contract, model naming and answer normalisation, without loading any weights.

`api.manager` is replaced by a stub, so routes run for real while no model loads. Anything that
needs inference is checked by running the server against real models.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from olla_jev import api, normalize
from olla_jev.manager import NotDownloaded, NotTrusted, check_limits
from olla_jev.names import parse, pick_gguf

NOUL = {"type": "noul", "noul": 0.9, "x_extra": 1}


class StubManager:
    def __init__(self):
        self.calls = []
        self.error = None

    def run(self, name, state, questions, keep_alive=None):
        self.calls.append((name, state, questions))
        if self.error:
            raise self.error
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = NOUL
            elif q["type"] == "choice":
                answers[qid] = {
                    "choice": next(iter(q["criteria"])),
                    "probabilities": {k: 1 / len(q["criteria"]) for k in q["criteria"]},
                }
            else:
                answers[qid] = {"probabilities": {str(i): 1 / len(q["criteria"]) for i in range(len(q["criteria"]))}}
        return SimpleNamespace(name="user/model"), {"answers": answers, "usage": {"input_tokens": 3}}

    def loaded(self):
        return []

    def unload_all(self):
        pass


@pytest.fixture
def stub(monkeypatch):
    m = StubManager()
    monkeypatch.setattr(api, "Manager", lambda: m)
    monkeypatch.setattr(api, "preload", None)
    return m


@pytest.fixture
def client(stub, tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    with TestClient(api.app, raise_server_exceptions=False) as c:
        yield c


def ask(client, questions, **body):
    return client.post("/v1/systemone", json={"state": "hello", "questions": questions, **body})


def test_response_matches_the_typesafe_schema(client):
    models = pytest.importorskip("typesafe_sdk._schemas.models")
    r = ask(
        client,
        {
            "n": {"type": "noul", "instructions": "y"},
            "c": {"type": "choice", "criteria": {"a": None, "b": "B"}},
            "s": {"type": "score", "criteria": ["low", "high"]},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    models.SystemOneResponse.model_validate(body)
    assert body["model"] == "user/model"
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 0}
    assert "x_extra" not in body["answers"]["n"]
    assert body["answers"]["s"]["legend"] == {"0": "low", "1": "high"}
    models.ModelMetadataList.model_validate(client.get("/v1/models").json())


def test_choice_options_without_description_reach_the_model(client, stub):
    ask(client, {"c": {"type": "choice", "criteria": {"a": None, "b": "B"}}})
    assert stub.calls[-1][2]["c"]["criteria"] == {"a": None, "b": "B"}


@pytest.mark.parametrize(
    "error,status,kind",
    [
        (NotDownloaded("x is not downloaded"), 404, "model_not_found"),
        (NotTrusted("x is not trusted"), 403, "model_not_trusted"),
        (ValueError("too many options"), 422, "value_error"),
    ],
)
def test_model_errors_keep_the_error_shape(client, stub, error, status, kind):
    stub.error = error
    r = ask(client, {"n": {"type": "noul"}})
    assert r.status_code == status
    assert r.json()["detail"][0]["type"] == kind


def test_under_two_criteria_is_422(client):
    assert ask(client, {"s": {"type": "score", "criteria": ["one"]}}).status_code == 422


def test_500_does_not_leak_the_exception(client, stub):
    stub.error = RuntimeError("secret detail")
    r = ask(client, {"n": {"type": "noul"}})
    assert r.status_code == 500 and "secret" not in r.text


def test_404_and_405_use_the_error_shape(client):
    assert client.get("/nope").json()["detail"][0]["type"] == "http_error"
    r = client.get("/v1/systemone")
    assert r.status_code == 405 and "allow" in r.headers


def test_every_preset_is_a_valid_request(client):
    for name, example in client.get("/ui/presets").json().items():
        assert ask(client, example["questions"]).status_code == 200, name


@pytest.mark.parametrize(
    "name,repo,tag",
    [
        ("SupersonicLabs/Julia-1", "SupersonicLabs/Julia-1", None),
        ("Mapika/decider-4b-GGUF:Q4_K_M", "Mapika/decider-4b-GGUF", "Q4_K_M"),
        ("hf.co/Mapika/decider-4b-GGUF:q8_0", "Mapika/decider-4b-GGUF", "q8_0"),
        ("https://huggingface.co/a/b:x.gguf", "a/b", "x.gguf"),
    ],
)
def test_parse_names(name, repo, tag):
    ref = parse(name)
    assert (ref.repo_id, ref.tag) == (repo, tag)


def test_parse_rejects_non_repo_names():
    with pytest.raises(ValueError):
        parse("decider-4b")


FILES = ["m-Q8_0.gguf", "m-Q4_K_M.gguf", "m-BF16.gguf", "config.json"]


def test_gguf_quant_selection_follows_ollama():
    assert pick_gguf(FILES, None) == "m-Q4_K_M.gguf"
    assert pick_gguf(FILES, "q8_0") == "m-Q8_0.gguf"
    assert pick_gguf(FILES, "m-BF16.gguf") == "m-BF16.gguf"
    assert pick_gguf(["m-Q8_0.gguf", "m-Q5_K_M.gguf"], None) == "m-Q5_K_M.gguf"
    with pytest.raises(ValueError):
        pick_gguf(FILES, "Q2_K")


def test_confidence_uses_typesafe_formulas():
    assert normalize.choice_confidence([0.5, 0.5]) == 0
    assert normalize.choice_confidence([1.0, 0.0, 0.0]) == 1
    assert normalize.score_confidence([0.0, 1.0, 0.0]) == 1
    a = normalize.answer({"type": "noul"}, {"noul": 0.75, "decision": "yes"})
    assert a == {"type": "noul", "noul": 0.75, "confidence": 0.5}


def test_per_question_model_errors_become_value_errors():
    with pytest.raises(ValueError):
        normalize.answers({"q": {"type": "noul"}}, {"q": {"type": "noul", "error": "max_length_exceeded"}})


def test_limits_are_checked_before_the_model_runs():
    questions = {"c": {"type": "choice", "criteria": {str(i): None for i in range(21)}}}
    with pytest.raises(ValueError, match="at most 20"):
        check_limits({"max_options": 20}, questions)
    check_limits({"max_options": 255}, questions)
    with pytest.raises(ValueError, match="at most 1 questions"):
        check_limits({"max_questions": 1}, {**questions, "d": {"type": "noul"}})


def test_fresh_machine_without_a_model_cache_lists_no_models(client, tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_MODELS", str(tmp_path / "never-created"))
    r = client.get("/v1/models")
    assert r.status_code == 200 and r.json() == {"models": []}
    assert client.get("/api/tags").json() == {"models": []}
