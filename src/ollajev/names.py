"""Model names, Ollama style: `<user>/<repo>`, `<user>/<repo>:<quant>` or `<user>/<repo>:<file.gguf>`.

An `hf.co/` or `huggingface.co/` prefix is accepted, so names copied from an Ollama command work,
and so are browser URLs of a repo or one of its files (`.../tree/main`, `.../blob/main/x.gguf`).
The quant is matched case-insensitively against the repo's file names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PREFIXES = (
    "https://huggingface.co/",
    "https://hf.co/",
    "http://huggingface.co/",
    "http://hf.co/",
    "hf.co/",
    "huggingface.co/",
)
# Ollama's order when a repo has no Q4_K_M: the first quant found, best compromise first.
QUANT_PREFERENCE = ["Q4_K_M", "Q4_K_S", "Q4_0", "IQ4_XS", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"]
_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")
_URL_PATH = re.compile(r"^([^/]+/[^/]+)/(?:blob|resolve|tree)/[^/]+(?:/(.*))?$")


@dataclass(frozen=True)
class Ref:
    repo_id: str
    tag: str | None = None  # quant or file name as written, None for the repo's default

    @property
    def name(self) -> str:
        return f"{self.repo_id}:{self.tag}" if self.tag else self.repo_id


def parse(name: str) -> Ref:
    text = name.strip()
    for prefix in PREFIXES:
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].split("?", 1)[0].split("#", 1)[0].rstrip("/")
            if m := _URL_PATH.match(text):
                file = (m.group(2) or "").rsplit("/", 1)[-1]
                text = f"{m.group(1)}:{file}" if file.lower().endswith(".gguf") else m.group(1)
            break
    repo_id, _, tag = text.partition(":")
    if not _REPO.match(repo_id) or ".." in repo_id or any(part.strip(".") == "" for part in repo_id.split("/")):
        raise ValueError(f"not a Hugging Face model name: {name!r} (expected <user>/<repo>[:<quant>])")
    return Ref(repo_id, tag or None)


def quant_of(filename: str) -> str | None:
    """`decider-4b-v2.1-Q4_K_M.gguf` or `laya_english_ud_q4_k_m.gguf` -> `Q4_K_M`."""
    m = re.search(r"[._-]((?:I?Q\d[\w]*)|BF16|F16|F32)\.gguf$", filename, re.IGNORECASE)
    return m.group(1).upper() if m else None


def pick_gguf(files: list[str], tag: str | None) -> str:
    """The one .gguf file `tag` names in a repo's file list, or the default quant when tag is None."""
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        raise ValueError("repo has no .gguf files")
    if tag is None:
        by_quant = {quant_of(f): f for f in ggufs}
        for q in QUANT_PREFERENCE:
            if q in by_quant:
                return by_quant[q]
        return sorted(ggufs, key=lambda f: f.lower())[0]
    if tag.lower().endswith(".gguf"):
        matches = [f for f in ggufs if f.rsplit("/", 1)[-1].lower() == tag.lower()]
    else:
        matches = [f for f in ggufs if quant_of(f) == tag.upper()]
    if len(matches) != 1:
        available = ", ".join(sorted({quant_of(f) or f for f in ggufs}))
        raise ValueError(f"no single file matches {tag!r}; available: {available}")
    return matches[0]
