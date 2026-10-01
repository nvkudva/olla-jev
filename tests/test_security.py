"""Trust gate, API key, Host check, name parsing, config safety and manager races. No weights load."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from olla_jev import admin, api, cli, config, manager, store
from olla_jev.names import parse


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    monkeypatch.delenv("OLLAJEV_API_KEY", raising=False)
    monkeypatch.setattr(api, "allowed_hosts", None)
    return tmp_path


@pytest.fixture
def client(home, monkeypatch):
    class Idle:
        def loaded(self):
            return []

        def unload_all(self):
            pass

    monkeypatch.setattr(api, "Manager", Idle)
    monkeypatch.setattr(api, "preload", None)
    with TestClient(api.app, raise_server_exceptions=False) as c:
        yield c


def fake_resolved(runs_code: bool = True):
    family = SimpleNamespace(name="julia", runs_repo_code=runs_code)
    return SimpleNamespace(
        repo_id="user/repo", revision="a" * 40, family=family, gguf=None, ref=SimpleNamespace(name="user/repo")
    )


def test_pull_never_trusts_over_http(client, monkeypatch):
    r = fake_resolved()
    monkeypatch.setattr(store, "resolve", lambda *a, **k: r)
    monkeypatch.setattr(store, "download", lambda r: pytest.fail("downloaded an untrusted repo"))
    body = client.post("/api/pull", json={"model": "user/repo", "stream": False, "trust": True})
    assert body.status_code == 400
    assert "--trust" in body.json()["error"]
    assert config.load().get("trusted", []) == []


def test_pull_of_the_same_repo_twice_is_refused(client, monkeypatch):
    r = fake_resolved(runs_code=False)
    monkeypatch.setattr(store, "resolve", lambda *a, **k: r)
    monkeypatch.setattr(store, "download", lambda r: None)
    monkeypatch.setattr(admin, "canonical", lambda r: r.repo_id)
    admin._pulling.add("user/repo")
    try:
        out = client.post("/api/pull", json={"model": "user/repo", "stream": False})
    finally:
        admin._pulling.discard("user/repo")
    assert out.status_code == 400 and "already being pulled" in out.json()["error"]


def test_pull_hides_unexpected_errors(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("/secret/path exploded")

    monkeypatch.setattr(store, "resolve", boom)
    out = client.post("/api/pull", json={"model": "user/repo", "stream": False})
    assert "secret" not in out.text


def test_api_key_is_required_when_set(client, monkeypatch):
    monkeypatch.setenv("OLLAJEV_API_KEY", "s3cret")
    assert client.get("/").status_code == 200  # health probe stays open
    assert client.get("/api/tags").status_code == 401
    assert client.get("/api/tags", headers={"authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/tags", headers={"authorization": "Bearer s3cret"}).status_code == 200


def test_no_key_means_open(client):
    assert client.get("/api/ps").status_code == 200


@pytest.mark.parametrize("host,status", [("localhost:8000", 200), ("[::1]:8000", 200), ("evil.example", 403)])
def test_host_header_is_checked_on_loopback(client, monkeypatch, host, status):
    monkeypatch.setattr(api, "allowed_hosts", frozenset({"localhost", "127.0.0.1", "[::1]"}))
    assert client.get("/api/ps", headers={"host": host}).status_code == status


def test_serve_refuses_a_public_bind_without_a_key(home, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.delenv("OLLAJEV_API_KEY", raising=False)
    args = cli.build_parser().parse_args(["serve", "--host", "0.0.0.0", "--no-browser"])
    with pytest.raises(SystemExit, match="OLLAJEV_API_KEY"):
        cli.cmd_serve(args)


@pytest.mark.parametrize("name", ["a/..", "../a", "./b", "https://evil.example/a/b", "a/b/c", "x"])
def test_parse_rejects(name):
    with pytest.raises(ValueError):
        parse(name)


@pytest.mark.parametrize("name", ["https://huggingface.co/u/r", "hf.co/u/r", "u/r"])
def test_parse_accepts_hf_forms(name):
    assert parse(name).repo_id == "u/r"


def test_corrupt_config_is_backed_up(home):
    (home / "config.json").write_text("{not json")
    assert config.load() == {}
    assert (home / "config.json.bad").read_text() == "{not json"


def test_concurrent_edits_do_not_lose_updates(home):
    def add(i):
        with config.edit() as data:
            data.setdefault("n", []).append(i)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(config.load()["n"]) == list(range(40))


def test_legacy_config_is_read_when_home_is_default(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy" / "config.json"
    legacy.parent.mkdir()
    legacy.write_text('{"default_model": "u/r"}')
    monkeypatch.delenv("OLLAJEV_HOME", raising=False)
    monkeypatch.setattr(config.Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    monkeypatch.setattr(config, "_legacy_path", lambda: legacy)
    assert config.load() == {"default_model": "u/r"}


class FakeAdapter:
    limits: dict = {}
    name = ""

    def system_one(self, state, questions):
        return {"answers": {}}


def test_run_retries_when_the_slot_was_unloaded(home, monkeypatch):
    mgr = manager.Manager.__new__(manager.Manager)
    dead = SimpleNamespace(
        adapter=FakeAdapter(), lock=threading.Lock(), closed=True, name="m", pinned=False, expires=0.0
    )
    live = SimpleNamespace(
        adapter=FakeAdapter(), lock=threading.Lock(), closed=False, name="m", pinned=False, expires=0.0
    )
    slots = iter([dead, live])
    mgr.get = lambda name, keep_alive=None: next(slots)  # type: ignore[method-assign]
    mgr._touch = lambda slot, keep_alive: None  # type: ignore[method-assign]
    slot, result = mgr.run("m", "state", {})
    assert slot is live and result == {"answers": {}}
