"""CLI parsing, service file generation and config locations. No server, no models."""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from olla_jev import cli, config, service


@pytest.mark.parametrize(
    "argv,func",
    [
        ([], "cmd_serve"),
        (["serve", "user/repo"], "cmd_serve"),
        (["pull", "a/b", "c/d:Q8_0", "--trust"], "cmd_pull"),
        (["ls"], "cmd_list"),
        (["service", "status"], "cmd_service"),
    ],
)
def test_commands_route_to_their_handlers(argv, func):
    args = cli.build_parser().parse_args(argv)
    assert args.func.__name__ == func


def test_pull_takes_several_models():
    args = cli.build_parser().parse_args(["pull", "a/b", "c/d:Q8_0", "--trust"])
    assert args.model == ["a/b", "c/d:Q8_0"] and args.trust


def test_unknown_service_action_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["service", "restart"])
    assert exc.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--version"])
    assert capsys.readouterr().out.startswith("olla-jev ")


def test_launchd_plist_is_valid_and_runs_serve(tmp_path):
    data = plistlib.loads(service.plist("/opt/bin/olla-jev", tmp_path).encode())
    assert data["Label"] == service.LABEL
    assert data["ProgramArguments"] == ["/opt/bin/olla-jev", "serve", "--no-browser"]
    assert data["RunAtLoad"] and data["KeepAlive"]
    assert data["StandardErrorPath"] == str(tmp_path / "service.err.log")


def test_plist_escapes_paths(tmp_path):
    data = plistlib.loads(service.plist("/Users/a&b/olla-jev", tmp_path).encode())
    assert data["ProgramArguments"][0] == "/Users/a&b/olla-jev"


def test_systemd_unit_quotes_the_executable():
    text = service.unit("/home/me/my tools/olla-jev")
    assert 'ExecStart="/home/me/my tools/olla-jev" serve --no-browser' in text
    assert "WantedBy=default.target" in text


def test_olla_jev_home_overrides_config_and_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAJEV_HOME", str(tmp_path))
    assert config.config_path() == tmp_path / "config.json"
    assert config.log_dir() == tmp_path / "logs"


def test_default_config_dir_is_the_os_folder(monkeypatch):
    monkeypatch.delenv("OLLAJEV_HOME", raising=False)
    assert config.config_dir().name == "olla-jev"
    assert Path.home() in config.config_dir().parents


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", ("127.0.0.1", 8000)),
        ("0.0.0.0", ("0.0.0.0", 8000)),
        ("127.0.0.1:9000", ("127.0.0.1", 9000)),
        ("http://localhost:9000/", ("localhost", 9000)),
        ("[::1]:9000", ("::1", 9000)),
    ],
)
def test_olla_jev_host(monkeypatch, value, expected):
    monkeypatch.setenv("OLLAJEV_HOST", value)
    assert config.host() == expected


@pytest.mark.parametrize("text,seconds", [("300", 300), ("5m", 300), ("1h", 3600), ("-1", -1), (30, 30)])
def test_keep_alive_durations(text, seconds):
    assert config.parse_duration(text) == seconds
