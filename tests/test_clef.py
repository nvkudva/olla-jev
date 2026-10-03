"""The clef adapter's request mapping, answer shaping and repo detection, without loading the 9B backbone."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ollajev import normalize
from ollajev._vendor.clef import joint_schema_model as clef_model
from ollajev.adapters import clef, detect

CLEF_FILES = [
    "config.json",
    "joint_head.safetensors",
    "joint_head_config.json",
    "joint_schema_model.py",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "tokenizer.json",
]
QUESTIONS = {
    "outage": {"type": "noul", "instructions": "Is a service down?"},
    "team": {"type": "choice", "instructions": "Who handles it?", "criteria": {"tech": "Bugs", "billing": None}},
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
}


class CharTokenizer:
    """One token per character, so token spans read back as the text they cover."""

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) for c in text])


def text(ids, span):
    return "".join(map(chr, ids[span[0] : span[1]]))


def test_detects_clef_repos_only():
    assert detect("Cloudflare/clef-flash", CLEF_FILES) is clef.FAMILY
    assert not clef.FAMILY.matches("Mapika/decider-2b", ["decider_config.json", "model.safetensors"])
    with pytest.raises(LookupError):  # a GGUF copy carries no joint head
        detect("bartowski/Cloudflare_clef-flash-GGUF", ["Cloudflare_clef-flash-Q4_K_M.gguf", "README.md"])
    mlx = [f for f in CLEF_FILES if f != "joint_schema_model.py"] + ["clef_mlx.py"]
    vllm = [*CLEF_FILES, "clef_vllm.py", "recipe.yaml"]
    fp8 = [*CLEF_FILES, "recipe.yaml"]
    for files in (mlx, vllm, fp8, [*CLEF_FILES, "clef_exl3.py"]):
        assert not clef.FAMILY.matches("someone/clef-copy", files)


def test_record_keeps_wire_questions_and_noul_defaults():
    rec = clef.record(
        {"msg": "hi"},
        {
            "a": {"type": "noul", "instructions": None, "criteria": {"true": "Yes", "false": None, "other": "x"}},
            "b": {"type": "noul", "criteria": None},
            "c": QUESTIONS["team"],
        },
    )
    assert rec["state"] == {"msg": "hi"}
    assert rec["questions"]["a"] == {"type": "noul", "criteria": {"true": "Yes"}}
    assert rec["questions"]["b"] == {"type": "noul", "criteria": {}}
    assert rec["questions"]["c"] == QUESTIONS["team"]
    assert clef_model.question_options(rec["questions"]["a"])[0] == ("true", "Yes")


def test_prompt_spans_cover_instructions_and_options():
    enc = clef_model.encode_record(CharTokenizer(), clef.record("Checkout is down", QUESTIONS))
    prompt = "".join(map(chr, enc.input_ids))
    assert prompt.startswith("<|im_start|>system\n")
    assert "STATE:\nCheckout is down\n\nSCHEMA FIELDS:\n" in prompt
    assert prompt.endswith("<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    noul, choice, score = enc.questions
    assert text(enc.input_ids, noul.question_span) == "Is a service down?"
    assert noul.option_ids == ("true", "false")
    assert choice.option_ids == ("billing", "tech")  # sorted by id
    assert text(enc.input_ids, choice.option_spans[0]) == '{"option_id":"billing"}'
    assert text(enc.input_ids, choice.option_spans[1]) == '{"description":"Bugs","option_id":"tech"}'
    assert text(enc.input_ids, score.question_span) == "urgency"  # no instructions: the id
    assert score.option_ids == ("0", "1", "2")


def test_head_scores_every_option():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    head = clef_model.JointSchemaHead(hidden_size=16, width=8, routing_layers=1, layers=1, heads=2, feedforward=16)
    enc = clef_model.encode_record(CharTokenizer(), clef.record("Checkout is down", QUESTIONS))
    batch = clef_model.collate_records([enc], 0, torch.device("cpu"))
    hidden = torch.randn(1, len(enc.input_ids), 16)
    with torch.inference_mode():
        logits = head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"], torch.randn(200, 16))[0]
    assert [len(x) for x in logits] == [2, 2, 3]


def test_answers_normalize_to_the_wire_shape():
    raw = {
        "outage": clef.answer("noul", {"true": 0.9, "false": 0.1}),
        "team": clef.answer("choice", {"billing": 0.2, "tech": 0.8}),
        "urgency": clef.answer("score", {"0": 0.1, "1": 0.2, "2": 0.7}),
    }
    out = normalize.answers(QUESTIONS, raw)
    assert out["outage"]["noul"] == 0.9
    assert out["team"]["choice"] == "tech"
    assert list(out["team"]["probabilities"]) == ["tech", "billing"]  # request order
    assert out["urgency"]["score"] == pytest.approx(1.6)
    assert out["urgency"]["legend"]["2"] == "Today"
