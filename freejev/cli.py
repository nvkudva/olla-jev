"""Command line entry point. Commands mirror Ollama's: serve, run, pull, list, ps, show, rm, stop, cp."""

from __future__ import annotations

import argparse
from importlib.metadata import version

COMMANDS: dict[str, str] = {
    "serve": "start the server",
    "run": "load a model and ask it questions interactively",
    "pull": "download a model",
    "list": "list downloaded models",
    "ps": "list loaded models",
    "show": "show a model's details and limits",
    "rm": "delete a downloaded model",
    "stop": "unload a running model",
    "cp": "copy a model under a new name",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="free-jev-server", description=__doc__)
    parser.add_argument("--version", action="version", version=version("free-jev-server"))
    sub = parser.add_subparsers(dest="command", metavar="command")
    for name, help_text in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    raise SystemExit(f"{args.command}: not implemented yet")
