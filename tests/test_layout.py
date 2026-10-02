"""Dependencies point inward: the core never imports the server or the front ends, the server never imports
the front ends."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "ollajev"


def imports(path: Path) -> set[str]:
    """Every ollajev module `path` imports, as a dotted name, relative imports resolved."""
    here = ["ollajev", *path.relative_to(PACKAGE).with_suffix("").parts]
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            base = here[: len(here) - node.level] if node.level else []
            module = ".".join([*base, *(node.module.split(".") if node.module else [])])
            found.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return {name for name in found if name.startswith("ollajev.")}


def modules(*exclude: str) -> list[Path]:
    return [
        p
        for p in PACKAGE.rglob("*.py")
        if not any(part in exclude for part in p.relative_to(PACKAGE).parts) and "_vendor" not in p.parts
    ]


@pytest.mark.parametrize("path", modules("ui", "server"), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_core_imports_neither_server_nor_ui(path):
    if path.name == "__main__.py":
        pytest.skip("the entry point starts the command line")
    bad = {m for m in imports(path) if m.startswith(("ollajev.ui", "ollajev.server"))}
    assert not bad, f"{path.name} imports {bad}"


@pytest.mark.parametrize("path", sorted((PACKAGE / "server").glob("*.py")), ids=lambda p: p.name)
def test_server_does_not_import_ui(path):
    bad = {m for m in imports(path) if m.startswith("ollajev.ui")}
    assert not bad, f"{path.name} imports {bad}"
