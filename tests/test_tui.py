"""The model manager screen: it mounts, lists the catalog, opens its dialogs and exits with the right answer."""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import DataTable

from ollajev import cli, tui
from ollajev.catalog import CATALOG


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.setenv("OLLAJEV_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(cli, "server_running", lambda: False)
    return tui.Models()


def drive(app, keys):
    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            table = app.query_one(DataTable)
            assert table.row_count == len(CATALOG)
            for key in keys:
                await pilot.press(key)
                await pilot.pause()
            return [type(screen).__name__ for screen in app.screen_stack][1:]

    return asyncio.run(go())


def test_serve_key_returns_true(app):
    drive(app, ["s"])
    assert app.return_value is True


def test_quit_key_returns_false(app):
    drive(app, ["q"])
    assert app.return_value is False


@pytest.mark.parametrize(("key", "screen"), [("o", "Options"), ("n", "Prompt"), ("a", "Prompt"), ("i", "Info")])
def test_keys_open_their_dialog_and_escape_closes_it(app, key, screen):
    assert drive(app, [key]) == [screen]
    assert drive(tui.Models(), [key, "escape"]) == []


def test_ask_needs_a_downloaded_model(app):
    assert drive(app, ["r"]) == []
