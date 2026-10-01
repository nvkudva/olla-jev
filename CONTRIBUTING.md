# Contributing

## Setup

```sh
git clone https://github.com/nvkudva/olla-jev.git
cd olla-jev
uv sync                      # Python 3.12 environment with the dev tools
uv run olla-jev --help
```

## Checks

Run these before opening a pull request. CI runs the same ones.

```sh
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
uv run pytest
```

`uv run pre-commit install` runs ruff on every commit.

The tests load no model weights. To check a change against a real model, run the server and send
it a request:

```sh
OLLAJEV_HOME=/tmp/olla-jev-dev uv run olla-jev serve SupersonicLabs/Julia-1 --port 8765
```

## Layout

| Path | What it holds |
|---|---|
| `src/olla_jev/cli.py` | command line |
| `src/olla_jev/api.py`, `admin.py` | HTTP routes: Jev API, management API |
| `src/olla_jev/manager.py` | loading, unloading and running models |
| `src/olla_jev/store.py`, `names.py` | model names, pinned downloads, trust |
| `src/olla_jev/adapters/` | one module per model family |
| `src/olla_jev/normalize.py` | answers in the TypeSafe shape |
| `src/olla_jev/_vendor/kev/` | kev's loader, vendored; do not edit except as noted in `VENDORED.md` |

## Adding a model family

Add a module under `src/olla_jev/adapters/` with a `FAMILY` object that implements the `Family`
protocol in `adapters/__init__.py` (`matches`, `allow_patterns`, `limits`, `load`), and register it
in `families()`. Set `runs_repo_code = True` if it imports Python from the model repo. Run the new
model end to end through `/v1/systemone` before adding it to `catalog.py`.

## Running tests

```sh
uv run pytest
```

## Lint and type check

```sh
uv run ruff check
uv run pyright
```

## Releasing

1. Bump `version` in `pyproject.toml`.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under the new version.
3. Commit, then tag `vX.Y.Z` and push the tag.

`.github/workflows/release.yml` checks the tag matches `pyproject.toml`, builds, publishes to PyPI
with trusted publishing, and creates the GitHub release.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:` …) and
add a line to `CHANGELOG.md` under `[Unreleased]` for user-visible changes.

## License

Contributions are made under the Apache-2.0 license, the same as the project. There is no CLA.
Add a `Signed-off-by` line (`git commit -s`) if you like; it is not required.
