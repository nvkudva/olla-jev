# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Jev / System One API (`GET /v1/models`, `POST /v1/systemone`), drop-in for `typesafe-sdk`.
- Adapters for seven model families: decider (PyTorch and llama.cpp GGUF), laya, Julia,
  open-jev, kev, Intern-Decision and Decision-1.0. Sixteen curated models under 4 GB.
- Ollama-style model names (`repo`, `repo:quant`, `repo:file.gguf`, `hf.co/` prefix), pinned to the
  commit of first download.
- Model manager: load on request, `keep_alive` unload, least-recently-used eviction.
- Trust prompt per repo and commit for families that run code from the model repo.
- Ollama-style management API (`/api/tags`, `ps`, `pull`, `show`, `delete`, `copy`, `stop`) and
  CLI (`serve`, `setup`, `run`, `pull`, `list`, `ps`, `show`, `rm`, `stop`, `cp`, `service`).
- Textual setup screen on first run; interactive `run`.
- Demo page with a model picker and per-model limits.
- `install.sh` / `install.ps1`, and a background service (launchd, systemd user unit).
