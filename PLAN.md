# free-jev-server — Plan

## Goal

- Local server that runs any supported System One (typed-decision) model from Hugging Face.
- Exposes the Jev / System One wire API (`GET /v1/models`, `POST /v1/systemone`), identical to laya-server and `api.typesafe.ai` v0.2.0.
- Drop-in for `typesafe-sdk` and `jev` decorators via `TYPESAFE_BASE_URL`.
- Ollama-like experience: pick a model, it downloads, it serves.

## Architecture

- Python 3.12, uv, FastAPI + uvicorn. Same stack as laya-server.
- Reuse laya-server pieces: request/response schema (`api.py`), port binding, logging, `/demo` web UI, `start.sh` / `start.ps1`.
- TUI: Textual. Runs on first launch or via `free-jev-server setup`.
- One model loaded per process. Requests serialized behind a lock.
- Adapter registry: one adapter per model family. Each adapter implements `load(snapshot_path, device, dtype)` and `predict(state, questions) -> answers` in the shared wire format.
- Family detection: curated catalog entry first, then repo files (`config.json` `architectures`/`model_type`, `adapter_config.json`, bundled package dirs). Unknown family is rejected with a clear error.
- Config saved to `~/.config/free-jev-server/config.toml` (model, revision SHA, device, dtype, host, port).

## Model formats

- Prefer each model's original published format and inference code.
- Most originals are PyTorch safetensors plus a Python package bundled inside the repo.
- Backends: PyTorch on MPS, CUDA or CPU.
- ONNX / GGUF / MLX / CoreML community ports are out of scope for v1. Possible later as optional backends.

## Curated catalog (TUI menu)

- `convaiinnovations/laya`, `laya-typed-decisions`, multilingual — `laya` package.
- `SupersonicLabs/Julia-1` — 144M mmBERT, multilingual, bundled `julia/` package.
- `com-kotobalabs/open-jev-deberta-v3-large` — 435M DeBERTa, bundled `typed_decisions/` package.
- `Mapika/decider-4b` (default), `decider-2b` — Qwen3.5, bundled `decider/` package.
- `jaredpalmer/kev-0.8b`, `kev-4b` — PEFT LoRA on Qwen3.5 base.
- Fallback: "paste repo id". Works only when the repo matches a supported family.

## Adapter families (v1)

- `laya` — `laya` PyPI package.
- `bundled-package` — imports the package shipped in the repo snapshot (Julia, open-jev, decider).
- `peft-pointer` — base model + LoRA adapter + pointer readout at option positions (kev).
- Later: `gliner2` (GLiNER2.5-Decide; no probabilities, needs a confidence policy), OpenThai-SystemOne, JEV-9B, NeoHorse.

## TUI first-run flow

- Choose model from curated catalog, or paste a repo id.
- Choose device (auto / mps / cuda / cpu) and dtype (fp32 / fp16 / bf16).
- Choose host and port (default 127.0.0.1, first free from 8000).
- For repos with bundled code: show repo, commit SHA and code files, require explicit trust.
- Download with progress, save config, start server.

## Security

- Bundled repo code is arbitrary Python that runs with user privileges.
- Pin the commit SHA at first download. Never auto-update.
- Require explicit trust per repo + SHA in the TUI. Store trusted pairs in config.
- Bind to 127.0.0.1 by default.

## Decisions

- Default model: `Mapika/decider-4b` (user confirmed 4B is fine).
- Curated menu + paste-repo-id fallback (user confirmed).

## Rejected alternatives

- Fully generic "any HF model" loader — no shared System One format exists; each family packs questions differently.
- ONNX-only runtime — originals are PyTorch with custom code; ONNX ports are second-hand and incomplete.
- GLiNER2.5-Decide in v1 — returns labels without probabilities; breaks the calibrated-confidence contract.
- `akhilaaa3/Jev-Omni` in catalog — 12B multimodal, too heavy for a default menu.
- `pngwn/system-one-*` — CC-BY-NC license.

## Revisions

### 2026-10-01 — Demo model switching and Ollama parity

- Demo page: model picker fed by `GET /v1/models`. Request `model` field selects the model. No more hardcoded `MODEL = "laya"`.
- Demo presets: own sample question sets, not imported from the `laya` package, so they work for every model.
- Per-model limits (max options, score levels, context length, languages) exposed by each adapter. API returns 422 on violation. Demo shows them as hints.
- Server is a long-running daemon. CLI is a thin client over HTTP, like Ollama.
- Models load on demand by the `model` field in a request. Idle models are unloaded after `keep_alive` (default 5m). `max_loaded_models` limits memory use.
- Lock is per model, not global.
- CLI commands mirror Ollama: `serve`, `run`, `pull`, `list`, `ps`, `show`, `rm`, `stop`, `cp`.
- Model names: short tags (`decider:4b`, `julia:1`, `kev:4b`) map to repo + pinned SHA. Full repo ids are also accepted.
- Env vars: `FREEJEV_HOST` (default `127.0.0.1:8000`), `FREEJEV_MODELS`, `FREEJEV_KEEP_ALIVE`, `FREEJEV_MAX_LOADED_MODELS`.
- Storage: Hugging Face cache by default, overridable by `FREEJEV_MODELS`.
- Admin API mirrors Ollama: `/api/tags`, `/api/ps`, `/api/pull` (streamed progress), `/api/show`, `/api/delete`. Wire API stays `/v1/models` and `/v1/systemone`.
- Textual TUI becomes the interactive layer: first-run setup, and `run` without a model. Scripts use plain CLI commands.
- Optional background service later: launchd (macOS), systemd (Linux), plus a menu-bar app.
- Distribution later: `uv tool install free-jev-server`, then Homebrew.

### 2026-10-01 — Inference code inspection (Tier 1)

Julia-1 (`SupersonicLabs/Julia-1`, 550 MiB fp32)
- Entry: `from julia import load_model`; `load_model(path, device, max_length=8192).predict(state=..., questions=...)`. Input is already the Jev `questions` map.
- Output: `type`, `probabilities`, `choice` / `score` / `noul`, `max_probability`. No `confidence`, `legend`, `action`. Adapter must add them.
- Limits: 2–20 options per question; 48 tokens per option; 8192 tokens total. State is a string only.
- Device: generic `torch.device`, so CPU / MPS / CUDA. Optional native Bend backend is CPU-only and not needed.
- Risk: its `pyproject.toml` pins `transformers>=5.0,<5.1`. We import from the snapshot, so the pin is not enforced. Must test against our transformers version.

open-jev-deberta (`com-kotobalabs/open-jev-deberta-v3-large`, 435M)
- Entry: `typed_decisions.open_jev.OpenJev.from_pretrained(path, device).decide(state, [questions])`. Input is a list with `options`, not the Jev `criteria` map. Adapter converts both ways.
- Output: per question `choice` / `score` / `noul`, `probabilities`, `confidence` (= max probability, not TypeSafe's formula). `noul` has no probabilities or confidence.
- Limits: `choice` 2–255, `score` 2–10, state cut to 256 tokens, 512 tokens total. English only. State is a string only.
- Device: auto CUDA / MPS / CPU. Plain transformers + safetensors. Lowest risk.

decider-4b (`Mapika/decider-4b`, 8.4 GB bf16, Qwen3.5-4B)
- Entry: `decider.infer.Decider(path, device).system_one(state, questions)`. Takes and returns the TypeSafe `/v1/systemone` shape directly, with TypeSafe's confidence formulas, `legend`, and string / dict / list state.
- Extra output fields: `x_p_max`, `certainty`, `level_fit`, `fit_mass`. Keep or strip; must not break typesafe-sdk.
- Limits: `choice` 2–255, `score` 2–10, state up to 32k tokens.
- Device: auto CUDA / MPS (fp16) / CPU. Ships its own MPS kernels for gated-delta attention. Card says MPS was not checked on v2.1.
- Optional dependency `flash-linear-attention` is CUDA-only; without it the model is several times slower.
- Best fit for the wire contract. Highest memory: needs about 9 GB free.

kev-4b (`jaredpalmer/kev-4b`, LoRA r16 on Qwen3.5-4B-Base)
- No inference code in the repo. Code lives in GitHub `jaredpalmer/kev` (Apache-2.0, `python -m kev.serve`), which serves `/v1/systemone`.
- Pointer head is `head.pt`, a pickle file. Loading it with `torch.load` can run arbitrary code. Load with `weights_only=True` only.
- Needs base model download (`Qwen/Qwen3.5-4B-Base`, about 8 GB) plus adapter.
- Card: slow on Mac, DeltaNet kernels have no MPS path (0.78 s per 5-question request on M5).
- Options: vendor the `kev` package pinned to a commit, or port its readout into our adapter.

Consequences
- Every Tier 1 model is a decision model with `noul` / `choice` / `score` and probabilities, so all four can serve `/v1/systemone` and the demo.
- Output normalisation layer needed: fill `confidence` (TypeSafe formulas from decider's `systemone.py`), `legend`, `action`, `usage` for models that lack them.
- Per-model limits differ a lot (Julia 20 options, open-jev 512 tokens). Adapter limits plus 422 validation are required, not optional.
- State as JSON object works only on decider. Other adapters serialise dict / list states to JSON text.
- Adapter order by risk: open-jev, Julia-1, decider-4b, kev-4b.
- Default model stays `decider-4b`, but must be verified on MPS first. Fallback default: Julia-1.

### 2026-10-01 — Phase 1 limited to models under 4 GB

- Phase 1 catalog: only models whose total download (weights plus any base model) is under 4 GB.
- Default model changes from `Mapika/decider-4b` (8.4 GB) to `Mapika/decider-2b` (3.76 GB). Same adapter and native `/v1/systemone` output.
- Phase 1 adapters: laya, Julia-1, open-jev, decider, kev, Intern-Decision, Decision-1.0.
- Phase 1 catalog: laya (en, typed-decisions, multilingual), Julia-1, open-jev-deberta-v3-large, decider-0.8b, decider-2b, kev-0.5b, kev-0.6b, kev-0.8b, Intern-Decision-0.8B, Decision-1.0-Kai-0.6B, Decision-1.0-Lex-0.6B.
- Deferred, small but awkward: GLiNER2.5-Decide and multi-Decide (no probabilities), OpenThai-SystemOne (custom modeling, Thai focus).
- Excluded, non-commercial license: bev-decider-0.4B, Muose-50M-Decision.
- Phase 2: models over 4 GB (decider-4b and larger, kev-4b and larger, Open-Jev, Intern-Decision-2B/4B, Decision-1.0 Sol/Nox/Lux, JEV-9B, NeoHorse, GLiNER2.5-Decide-1B), plus memory checks before load.
- Phase 2 addition: `bespokelabs/Bespoke-Nimble-9B` (+v2). LoRA on Qwen3.5-9B, ~18 GB total. Uses JSON-schema `enum` fields, not Jev `questions`, so it needs a mapping. Card targets CUDA; Mac runs through a separate MLX loader on GitHub.

### 2026-10-01 — Quantized variants under 4 GB

- `Mapika/decider-4b-GGUF` is official (same author). Q4_K_M is 2.7 GB, about 0.2 points lower in-task accuracy than bf16. Q8_0 is 4.5 GB.
- It runs through the author's `decide_gguf.py` on llama-cpp-python (Metal on Apple Silicon). Same option-letter readout and temperatures, so `/v1/systemone` output is unchanged.
- llama.cpp also avoids the slow PyTorch fallback for Qwen3.5 linear attention on MPS.
- Decision: add a llama.cpp backend to the decider adapter in phase 1.
- Default model returns to decider-4b, as `decider-4b` Q4_K_M GGUF (2.7 GB). User earlier accepted 4B; the 4 GB limit now holds.
- `Mapika/decider-2b-GGUF` (official) Q4_K_M 1.2 GB and Q8_0 2.0 GB join the catalog.
- `TokenRhythm/NeoHorse-Jev-4B-GGUF` is official with its own runtime; Q4_K_M 3.3 GB. Phase 2 candidate.
- Plain GGUF files in Ollama or llama-server give a text model only; the decision readout is required. Community GGUFs without readout code are not usable as-is.
- Rejected: community kev GGUFs (no pointer head), kev MLX 8-bit (4.4 GB, over limit), JEV-9B Q2_K (3.8 GB but head missing and heavy quality loss), community Intern-Decision GGUFs (readout unknown).
- Phase 2 candidate: `onnx-community/kev-4b-ONNX` q4 (~2.3 GB) on ONNX Runtime, if it carries the pointer head.

### 2026-10-01 — Model tag separator

- Model tags use `-`, not `:`. Examples: `decider-4b`, `julia-1`, `kev-0.6b`, `laya-multilingual`.
- Tags are whole lookup keys in the catalog, never split into family and variant.
- A bare family name (`decider`, `laya`, `julia`) is an alias for that family's default variant.

Phase 1 tags: `decider-4b` (default, Q4_K_M GGUF), `decider-2b`, `decider-2b-q8`, `decider-2b-bf16`, `decider-0.8b`, `laya`, `laya-multilingual`, `laya-typed-decisions`, `julia-1`, `open-jev-large`, `kev-0.5b`, `kev-0.6b`, `kev-0.8b`, `intern-decision-0.8b`, `decision1-kai-0.6b`, `decision1-lex-0.6b`.

### 2026-10-01 — Model names are Hugging Face repo ids (replaces short tags)

- No short tags and no aliases. A model name is the exact Hugging Face repo id, e.g. `SupersonicLabs/Julia-1`.
- Repos with several quantized files take a quant suffix: `<repo>-<quant>`, e.g. `Mapika/decider-4b-GGUF-Q4_K_M`.
- The suffix is valid only when it matches exactly one weight file in that repo. Otherwise the name is treated as a plain repo id.
- An exact file path also works: `<repo>/<file>`, e.g. `Mapika/decider-4b-GGUF/decider-4b-v2.1-Q4_K_M.gguf`.
- A multi-quant repo named without a suffix uses Q4_K_M.
- The `model` field in `/v1/systemone` and every CLI command take the same name.

Phase 1 names: `Mapika/decider-4b-GGUF-Q4_K_M` (default), `Mapika/decider-2b-GGUF-Q4_K_M`, `Mapika/decider-2b-GGUF-Q8_0`, `Mapika/decider-2b`, `Mapika/decider-0.8b`, `convaiinnovations/laya`, `convaiinnovations/laya-multilingual`, `convaiinnovations/laya-typed-decisions`, `SupersonicLabs/Julia-1`, `com-kotobalabs/open-jev-deberta-v3-large`, `jaredpalmer/kev-0.5b`, `jaredpalmer/kev-0.6b`, `jaredpalmer/kev-0.8b`, `internlm/Intern-Decision-0.8B`, `llm-semantic-router/Decision-1.0-Kai-0.6B`, `llm-semantic-router/Decision-1.0-Lex-0.6B`.

### 2026-10-01 — Quant selection follows Ollama (replaces `-<quant>` suffix)

- Syntax: `<repo>:<quant>`, as Ollama does for `hf.co/<user>/<repo>:<quant>`. Example: `Mapika/decider-4b-GGUF:Q4_K_M`.
- Quant name is case-insensitive (`:q4_k_m` works).
- Full file name also works as the tag: `Mapika/decider-4b-GGUF:decider-4b-v2.1-Q4_K_M.gguf`.
- No tag: Q4_K_M when present, otherwise one reasonable quant from the repo (prefer Q4, then Q5, then Q8).
- Optional `hf.co/` or `huggingface.co/` prefix is accepted, so names copied from Ollama snippets work.
- Repos with a single weight set take no tag: `SupersonicLabs/Julia-1`.
- `:` is safe as separator: Hugging Face repo ids cannot contain `:`.

Phase 1 names: `Mapika/decider-4b-GGUF:Q4_K_M` (default), `Mapika/decider-2b-GGUF:Q4_K_M`, `Mapika/decider-2b-GGUF:Q8_0`; all other names unchanged.

### 2026-10-01 — Phase 1 implementation decisions

- decider: uses the pinned `decider-ai` PyPI release (same code the repos bundle, plus the GGUF engine), so no repo code is imported. GGUF runs on llama.cpp with Metal.
- kev: vendored kev's loader (`api.py`, `model.py`, `checkpoint.py`, `device.py`) from GitHub at commit 90512f1 into `freejev/_vendor/kev`, instead of porting the readout. Reason: the readout depends on kev's hybrid-cache prefix logic; a port would diverge. kev's own package pins `torch<2.9`, so it cannot be a dependency. `head.pt` loads with `weights_only=True`.
- Julia-1: its fast encoder path calls a ModernBERT private method removed in transformers 5.1; the adapter disables that patch. Runs on CPU because its runtime moves inputs only for CUDA.
- Decision-1.0 Lex is fine-tuned from Kai (Vela encoder), not Qwen3.5 as first noted.
- Families that import repo code (Julia, open-jev, Intern-Decision, Decision-1.0) need trust per repo + commit, stored in config.
- Config is JSON (`~/.config/free-jev-server/config.json`), not TOML: it holds nested pins, trust and aliases, and the stdlib writes JSON.
- Confidence is always computed with TypeSafe's formulas in `normalize.py`, whatever the model reports, so it means the same across models. Noul confidence is the two-option choice formula.
- Default aliases `jev-latest` and `laya` mean the default model, for stock Jev / laya-server clients.
- The model `serve` preloads stays loaded (pinned); others follow keep_alive. Max loaded models defaults to 1.
- `cp` creates a name alias in config; it copies no files.
- `serve` scans for a free port from 8000 unless a port is given, and records the bound URL so other commands find it.

### 2026-10-01 — Renamed to olla-jev, src layout

- Project, command and distribution renamed from `free-jev-server` to `olla-jev`; import package `olla_jev` under `src/` (PyPA src layout).
- Environment variables renamed from `FREEJEV_*` to `OLLAJEV_*`.
- Rejected `ollama-jev`: "Ollama" is another company's product name, and Ollama now ships its own `/v1/systemone` (PR #18606, 2026-09-28), so the name would read as official. README states no affiliation.
- Earlier sections of this file use the old names.

### 2026-10-01 — Standard Python tool structure

- Install: `install.sh` / `install.ps1` replace `start.sh` / `start.ps1`. They install uv if missing, then `uv tool install` from a checkout, git URL or PyPI. `--service`, `--uninstall`.
- Developers use plain `uv sync` / `uv run`; no wrapper script.
- `olla-jev service install|uninstall|status|logs`: launchd agent `com.olla-jev.server` on macOS, systemd user unit `olla-jev.service` on Linux. Windows has no service; run `olla-jev serve`.
- Config and logs in OS folders via platformdirs; `OLLAJEV_HOME` overrides both. Models stay in the Hugging Face cache.
- Dev tools in a PEP 735 dependency group: pytest, ruff, pyright, typesafe-sdk, pre-commit.
- CI (GitHub Actions): ruff, pyright, pytest on macOS/Linux/Windows, wheel build. PyPI publishing waits for the GitHub repo.
- Rejected: an `init` command (means "create a project" in other tools; first-run setup is `olla-jev setup`). Rejected: click/typer (argparse covers the CLI).
- `PLAN.md` / `TODO.md` stay at the repo root.
