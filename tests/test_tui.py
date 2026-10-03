"""The model manager screen: it mounts, lists the catalog, opens its dialogs and exits with the right answer."""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import DataTable

from ollajev import client
from ollajev.catalog import CATALOG
from ollajev.ui import tui


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.setenv("OLLAJEV_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(client, "server_running", lambda: False)
    monkeypatch.setattr(tui, "variants", lambda repo: [])  # offline: the catalog's quants are not listed
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


@pytest.mark.parametrize(("key", "screen"), [("o", "Options"), ("n", "AddModel"), ("a", "Prompt"), ("i", "Info")])
def test_keys_open_their_dialog_and_escape_closes_it(app, key, screen):
    assert drive(app, [key]) == [screen]
    assert drive(tui.Models(), [key, "escape"]) == []


def test_ask_needs_a_downloaded_model(app):
    assert drive(app, ["r"]) == []


def quants(repo):
    return [tui.store.Variant(f"{repo}:{q}", q, n) for q, n in (("Q4_K_M", 2_700_000_000), ("Q8_0", 4_500_000_000))]


def test_add_model_lists_every_quant_and_returns_the_picked_one(app, monkeypatch):
    hits = [tui.store.Hit("u/ok-GGUF", 1200, "decider"), tui.store.Hit("u/no-GGUF", 5, None)]
    monkeypatch.setattr(tui.store, "search", lambda query, limit: hits)
    monkeypatch.setattr(tui, "variants", quants)
    picked = []

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            app.push_screen(tui.AddModel(), picked.append)
            await pilot.pause()
            await pilot.press(*"ok", "enter")
            await pilot.pause(0.5)
            results = app.screen.query_one("#results", DataTable)
            rows = [results.get_row_at(i) for i in range(results.row_count)]
            assert rows[0] == ["u/ok-GGUF:Q4_K_M", "2.7 GB", "1.2k", "✓ decider"]
            assert [r[0] for r in rows] == ["u/ok-GGUF:Q4_K_M", "u/ok-GGUF:Q8_0", "u/no-GGUF:Q4_K_M", "u/no-GGUF:Q8_0"]
            await pilot.press("down", "enter")
            await pilot.pause()

    asyncio.run(go())
    assert picked == ["u/ok-GGUF:Q8_0"]


def test_the_model_list_shows_every_quant_of_a_catalog_gguf_repo(app, monkeypatch):
    monkeypatch.setattr(tui, "variants", quants)

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause(0.3)
            return list(app.names)

    names = asyncio.run(go())
    repo = next(e.name for e in CATALOG if ":" in e.name).partition(":")[0]
    assert f"{repo}:Q4_K_M" in names and f"{repo}:Q8_0" in names
    assert len(names) == len(set(names))


def test_escape_cancels_a_running_download(app, monkeypatch):
    from types import SimpleNamespace

    r = SimpleNamespace(repo_id="u/r", family=SimpleNamespace(runs_repo_code=False), ref=SimpleNamespace(name="u/r"))
    started = {"set": False}

    def download(r, cancel):
        started["set"] = True
        cancel.wait(10)
        raise tui.store.Cancelled

    def resolve(name, online=True):
        if not online:  # the model list asks offline whether each is downloaded
            raise LookupError(name)
        return r

    monkeypatch.setattr(tui.store, "resolve", resolve)
    monkeypatch.setattr(tui.store, "download", download)
    monkeypatch.setattr(tui, "canonical", lambda r: r.repo_id)

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            assert app.query_one(DataTable).row_count == len(CATALOG)
            await pilot.press("p")
            for _ in range(100):
                if started["set"]:
                    break
                await pilot.pause(0.05)
            else:
                pytest.fail(f"download never started: selected={app.selected()!r} busy={app.busy}")
            await pilot.press("escape")
            for _ in range(100):
                await pilot.pause(0.05)
                if not app.busy:
                    return
            pytest.fail("download was not cancelled")

    asyncio.run(go())
