"""The curated phase 1 models: every one under 4 GB and checked to answer /v1/systemone."""

from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_MODEL


@dataclass(frozen=True)
class Entry:
    name: str
    size_gb: float
    languages: str
    description: str


CATALOG: list[Entry] = [
    Entry(DEFAULT_MODEL, 2.7, "English", "Qwen3.5-4B decider, 4-bit GGUF on llama.cpp"),
    Entry("Mapika/decider-2b-GGUF:Q4_K_M", 1.2, "English", "Qwen3.5-2B decider, 4-bit GGUF on llama.cpp"),
    Entry("Mapika/decider-2b-GGUF:Q8_0", 2.0, "English", "Qwen3.5-2B decider, 8-bit GGUF on llama.cpp"),
    Entry("Mapika/decider-2b", 3.76, "English", "Qwen3.5-2B decider, bf16 on PyTorch"),
    Entry("Mapika/decider-0.8b", 1.5, "English", "Qwen3.5-0.8B decider, bf16 on PyTorch"),
    Entry("convaiinnovations/laya", 0.85, "English", "Laya, ModernBERT-large, general purpose"),
    Entry("convaiinnovations/laya-multilingual", 0.68, "100+ languages", "Laya, mmBERT-base"),
    Entry("convaiinnovations/laya-typed-decisions", 0.85, "English", "Laya tuned for agent traces, support, invoices"),
    Entry("SupersonicLabs/Julia-1", 0.57, "Multilingual", "Julia 1, mmBERT-small, 2-20 options"),
    Entry("com-kotobalabs/open-jev-deberta-v3-large", 1.74, "English", "open-jev, DeBERTa-v3-large, 512 tokens"),
    Entry("jaredpalmer/kev-0.5b", 1.0, "English", "Kev, LoRA + pointer head on Qwen2.5-0.5B"),
    Entry("jaredpalmer/kev-0.6b", 1.2, "English", "Kev, LoRA + pointer head on Qwen3-0.6B"),
    Entry("jaredpalmer/kev-0.8b", 1.7, "English", "Kev, LoRA + pointer head on Qwen3.5-0.8B"),
    Entry("internlm/Intern-Decision-0.8B", 1.7, "Multilingual", "Intern-Decision, Qwen3.5-0.8B"),
    Entry("llm-semantic-router/Decision-1.0-Kai-0.6B", 2.28, "English", "Decision-1.0 Kai, Vela encoder"),
    Entry(
        "llm-semantic-router/Decision-1.0-Lex-0.6B",
        2.28,
        "English",
        "Decision-1.0 Lex, Kai tuned for support, invoices, agent traces",
    ),
]
