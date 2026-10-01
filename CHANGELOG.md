# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Security

- `POST /api/pull` no longer accepts `trust`; trusting a repo's Python code is CLI-only.
- `OLLAJEV_API_KEY` requires a bearer token on every API route. Listening on a non-loopback
  address without it is refused. On loopback, only `localhost`, `127.0.0.1` and `[::1]` are
  accepted as Host (DNS-rebinding guard).
- Model names reject `.`/`..` segments and foreign URL hosts.
- Requests over `OLLAJEV_MAX_BODY_BYTES` (8 MiB) get 413. The demo page sends a Content-Security-Policy.

### Changed

- `ollajev setup` (alias `tui`) is now a model manager: download, switch the default, ask, unload, delete,
  alias, inspect and serve from one screen. It no longer shows buttons and form fields.

- Config and logs now live in `~/.ollajev` (override with `OLLAJEV_HOME`). A config in the old OS
  folder is read once and moved on the next save.
- Dependencies use compatible ranges in `pyproject.toml`; `uv.lock` keeps exact versions.
- Installers pin to a release tag.

### Fixed

- Config writes are locked and atomic; a corrupt `config.json` is moved to `config.json.bad`.
- A request can no longer run on a model the reaper just unloaded; eviction reads the slot table
  under its lock.
- Two pulls of the same repo cannot run at once; unexpected pull errors are logged, not sent to the client.
- Repo-code imports are serialised.
- Removing one quant no longer deletes a blob another revision still uses.
- A failed download no longer leaves a pin.
- A bad `OLLAJEV_HOST`, `OLLAJEV_KEEP_ALIVE` or `OLLAJEV_MAX_LOADED_MODELS` gives a one-line error naming the variable.

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
