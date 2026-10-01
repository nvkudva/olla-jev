#!/bin/sh
# Install olla-jev as a command on your PATH, optionally as a background service.
#
#   curl -fsSL https://raw.githubusercontent.com/nvkudva/olla-jev/main/install.sh | sh
#   curl -fsSL https://raw.githubusercontent.com/nvkudva/olla-jev/main/install.sh | sh -s -- --service
#   ./install.sh                 # from a checkout: installs that checkout
#   ./install.sh --uninstall
#
# Options:
#   --service         also run the server in the background at login (macOS launchd, Linux systemd)
#   --source <spec>   what to install: a path, a git URL, or a PyPI requirement
#   --uninstall       remove the service and the command (your config and models are kept)
set -eu

REPO_URL="git+https://github.com/nvkudva/olla-jev"
SERVICE=0
UNINSTALL=0
SOURCE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --service) SERVICE=1 ;;
    --uninstall) UNINSTALL=1 ;;
    --source) shift; SOURCE="${1:?--source needs a value}" ;;
    -h|--help) sed -n '2,13p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '==> %s\n' "$*"; }

export PATH="$HOME/.local/bin:$PATH"

if [ "$UNINSTALL" = 1 ]; then
  if command -v olla-jev >/dev/null 2>&1; then
    olla-jev service uninstall >/dev/null 2>&1 || true
  fi
  if command -v uv >/dev/null 2>&1; then
    uv tool uninstall olla-jev >/dev/null 2>&1 && say "removed the olla-jev command" || say "olla-jev was not installed"
  fi
  say "kept your config and downloaded models; delete them by hand if you want the space back:"
  echo "    config  macOS ~/Library/Application Support/olla-jev   Linux ~/.config/olla-jev"
  echo "    models  ~/.cache/huggingface/hub (shared with other Hugging Face tools)"
  exit 0
fi

# A checkout installs itself; a piped install takes the GitHub repo.
if [ -z "$SOURCE" ]; then
  here=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || true)
  if [ -n "$here" ] && [ -f "$here/pyproject.toml" ] && grep -q '^name = "olla-jev"' "$here/pyproject.toml"; then
    SOURCE="$here"
  else
    SOURCE="$REPO_URL"
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v uv >/dev/null 2>&1 || { echo "uv install failed; see https://docs.astral.sh/uv/" >&2; exit 1; }
fi

say "installing olla-jev from $SOURCE (Python 3.12, its own environment)"
uv tool install --python 3.12 --force "$SOURCE"

case ":$PATH:" in
  *":$(uv tool dir --bin):"*) ;;
  *) uv tool update-shell >/dev/null 2>&1 || true
     say "added $(uv tool dir --bin) to your PATH; open a new terminal to use olla-jev" ;;
esac

if [ "$SERVICE" = 1 ]; then
  say "installing the background service"
  "$(uv tool dir --bin)/olla-jev" service install
fi

say "done. Next:"
if [ "$SERVICE" = 1 ]; then
  echo "    olla-jev pull Mapika/decider-4b-GGUF:Q4_K_M   # download the default model"
  echo "    olla-jev run                                  # ask it questions"
else
  echo "    olla-jev            # first run: pick a model, then serve"
fi
