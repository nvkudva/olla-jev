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
    monkeypatch.setattr(tui.dialogs, "variants", lambda repo: [])  # offline: the catalog's quants are not listed
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


@pytest.mark.parametrize("key", ["q", "ctrl+q"])
def test_quit_keys_return_false(app, key):
    drive(app, [key])
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
    monkeypatch.setattr(tui.dialogs, "variants", quants)
    picked = []

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            app.push_screen(tui.dialogs.AddModel(), picked.append)
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
    monkeypatch.setattr(tui.dialogs, "variants", quants)

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause(0.3)
            return list(app.names)

    names = asyncio.run(go())
    repo = next(e.name for e in CATALOG if ":" in e.name).partition(":")[0]
    assert f"{repo}:Q4_K_M" in names and f"{repo}:Q8_0" in names
    assert len(names) == len(set(names))


def test_escape_cancels_a_running_download(app, monkeypatch):
    import threading
    from types import SimpleNamespace

    r = SimpleNamespace(repo_id="u/r", family=SimpleNamespace(runs_repo_code=False), ref=SimpleNamespace(name="u/r"))
    started = threading.Event()

    def resolve(name, online=True):
        if not online:  # the model list asks offline whether each is downloaded
            raise LookupError(name)
        return r

    def download(r, cancel):
        started.set()
        cancel.wait(10)
        raise tui.store.Cancelled

    monkeypatch.setattr(tui.store, "resolve", resolve)
    monkeypatch.setattr(tui.store, "download", download)
    monkeypatch.setattr(tui, "canonical", lambda r: r.repo_id)

    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.press("p")
            assert await asyncio.to_thread(started.wait, 5)
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            assert not app.busy

    asyncio.run(go())


@pytest.mark.parametrize("key", ["x", "u", "s", "b"])
def test_actions_wait_while_a_model_loads(app, monkeypatch, key):
    monkeypatch.setattr(tui.store, "remove", lambda resolved: pytest.fail("removed during a load"))
    monkeypatch.setattr(tui.service, "status", lambda: pytest.fail("service touched during a load"))
    app.loading = "some/model"
    assert drive(app, [key]) == []
    assert app.return_value is None


def test_quit_during_a_download_asks_first(app):
    async def go():
        async with app.run_test(size=(120, 36)) as pilot:
            app.downloading = True
            await pilot.press("q")
            await pilot.pause()
            assert type(app.screen).__name__ == "Confirm"
            await pilot.press("n")
            await pilot.pause()
            assert app.return_value is None
            await pilot.press("q")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
        return app.return_value

    assert asyncio.run(go()) is False
