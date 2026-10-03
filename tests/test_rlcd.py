"""The rlcd adapter's detection, prompts, calibration and answer shaping, without loading weights."""

from __future__ import annotations

import math

import pytest

from ollajev import adapters, normalize
from ollajev.adapters import rlcd

FILES = [
    "README.md",
    "bundle_manifest.json",
    "calibrator.json",
    "config.json",
    "model.onnx",
    "model.safetensors",
    "model_fp16.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "train_manifest.json",
]


def test_detects_rlcd_and_only_rlcd():
    assert adapters.detect("heman10x/rlcd-modernbert-151m", FILES) is rlcd.FAMILY
    assert not rlcd.FAMILY.matches("x/y", ["config.json", "model.safetensors", "tokenizer.json"])
    assert not rlcd.FAMILY.matches("x/y", ["rl_agent_config.json", "model.safetensors"])
    assert "model.safetensors" in rlcd.FAMILY.allow_patterns(None)
    assert not any("onnx" in p for p in rlcd.FAMILY.allow_patterns(None))


def test_prompts_follow_the_reference_contract():
    text, keys = rlcd.build(
        "ctx", "intent", {"type": "choice", "instructions": "Why?", "criteria": {"a": "A", "b": None}}
    )
    assert (
        text == "<<LABEL>>It is A<<LABEL>>It is b<<LABEL>>insufficient evidence<<SEP>>Question: Why?\n\nContext:\nctx"
    )
    assert keys == ["a", "b"]
    text, keys = rlcd.build("ctx", "q", {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]})
    assert text.startswith("<<LABEL>>low (Value: 0)<<LABEL>>high (Value: 1)<<LABEL>>insufficient evidence<<SEP>>")
    assert keys == ["0", "1"]
    text, keys = rlcd.build("ctx", "is_refund", {"type": "noul"})
    assert text == (
        "<<LABEL>>true: is refund<<LABEL>>false: not is refund<<LABEL>>insufficient evidence<<SEP>>"
        "Context:\nctx\n\nEvaluate proposition: is refund"
    )
    assert keys == ["true", "false"]


def test_temperature_is_per_candidate_count_with_a_global_fallback():
    cal = {"temperature": 2.8, "log_temperature": math.log(2.8), "per_k": {"3": 5.0}}
    assert rlcd.temperature(cal, 3) == 5.0
    assert rlcd.temperature(cal, 8) == pytest.approx(2.8)


def test_answers_leave_out_the_abstention_slot():
    noul = rlcd.shape("noul", ["true", "false"], [2.0, 0.0, 2.0], 1.0)
    assert noul["noul"] == pytest.approx(1 / (1 + math.exp(-2)))
    assert noul["p_abstain"] == pytest.approx(math.exp(2) / (2 * math.exp(2) + 1))
    choice = rlcd.shape("choice", ["a", "b", "c"], [0.0, 3.0, 0.0, 0.0], 1.0)
    assert choice["choice"] == "b" and sum(choice["probabilities"].values()) == pytest.approx(1)
    score = rlcd.shape("score", ["0", "1", "2"], [0.0, 0.0, 9.0, 0.0], 1.0)
    out = normalize.answer({"type": "score", "criteria": ["a", "b", "c"]}, score)
    assert out["score"] > 1.9 and set(out) == {"type", "score", "confidence", "legend", "probabilities"}
