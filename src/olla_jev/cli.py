"""Run System One decision models from Hugging Face behind the Jev API.

Commands mirror Ollama's: serve, run, pull, list, ps, show, rm, stop, cp. With no command it serves,
and the first run opens a setup screen to pick a model.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from importlib.metadata import version
from typing import Any

from . import config, store
from .manager import canonical, default_model, lookup

DEFAULT_LOG = "server.log"


# ---- talking to a running server ------------------------------------------------------------------


def server_url() -> str:
    """OLLAJEV_HOST when set, else the address the last `serve` bound, else the default."""
    import os

    if os.environ.get("OLLAJEV_HOST"):
        host, port = config.host()
        return f"http://{url_host(host)}:{port}"
    return config.load().get("server_url") or f"http://127.0.0.1:{config.DEFAULT_PORT}"


def call(method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 600) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(server_url() + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read() or b"{}")
        raise SystemExit(f"error: {payload.get('error') or payload.get('detail') or exc}") from None


def server_running() -> bool:
    try:
        urllib.request.urlopen(server_url() + "/", timeout=0.5).close()
        return True
    except (urllib.error.URLError, OSError):
        return False


def need_server() -> None:
    if not server_running():
        raise SystemExit(f"could not connect to olla-jev at {server_url()}; start it with: olla-jev serve")


# ---- model management -----------------------------------------------------------------------------


def confirm_trust(r: store.Resolved, assume_yes: bool) -> None:
    """Families that import Python from the model repo run it with your privileges. Ask once per commit."""
    if store.is_trusted(r):
        return
    code = sorted(f for f in r.files if f.endswith(".py"))
    print(f"\n{canonical(r)} runs Python code from its Hugging Face repo, with your user's privileges.")
    print(f"  commit  {r.revision}")
    print(f"  review  https://huggingface.co/{r.repo_id}/tree/{r.revision}")
    if code:
        print(f"  files   {', '.join(code[:8])}{' …' if len(code) > 8 else ''}")
    if not assume_yes:
        if not sys.stdin.isatty():
            raise SystemExit("not trusted; review the code, then pull again with --trust")
        if input("\nTrust this exact commit? Type 'yes': ").strip() != "yes":
            raise SystemExit("not trusted; nothing downloaded")
    store.trust(r)


def pull(name: str, trust: bool = False) -> store.Resolved:
    r = store.resolve(lookup(name))
    confirm_trust(r, trust)
    print(f"==> pulling {canonical(r)} ({r.family.name}) at {r.revision[:12]}", flush=True)
    store.download(r)
    prefetch = getattr(r.family, "prefetch", None)
    if prefetch:
        print("==> pulling base model", flush=True)
        prefetch(store.local_path(r))
    print(f"==> success: {canonical(r)}", flush=True)
    return r


def cmd_pull(args: argparse.Namespace) -> None:
    for name in args.model:
        pull(name, args.trust)


def cmd_list(args: argparse.Namespace) -> None:
    from .admin import tags

    rows = tags()
    if not rows:
        print("no models downloaded; try: olla-jev pull " + config.DEFAULT_MODEL)
        return
    default = canonical_or(default_model())
    width = max(len(m["name"]) for m in rows)
    print(f"{'NAME':<{width}}  {'FAMILY':<16} {'SIZE':>8}  MODIFIED")
    for m in rows:
        mark = " *" if m["name"] == default else ""
        print(f"{m['name']:<{width}}  {m['details']['family']:<16} {m['size'] / 1e9:>6.2f} GB  {m['modified_at'][:10]}{mark}")


def canonical_or(name: str) -> str:
    try:
        return canonical(store.resolve(name, online=False))
    except (LookupError, ValueError):
        return name


def cmd_show(args: argparse.Namespace) -> None:
    try:
        r = store.resolve(lookup(args.model), online=False)
    except LookupError as exc:
        raise SystemExit(str(exc)) from None
    print(f"  model        {canonical(r)}")
    print(f"  family       {r.family.name}")
    print(f"  revision     {r.revision}")
    if r.gguf:
        print(f"  file         {r.gguf}")
    print(f"  released     {store.released(r.repo_id) or '-'}")
    print(f"  repo code    {'yes, ' + ('trusted' if store.is_trusted(r) else 'NOT trusted') if r.family.runs_repo_code else 'no'}")
    for key, value in r.family.limits(r).items():
        print(f"  {key:<12} {value}")
    print(f"  path         {store.local_path(r)}")


def cmd_rm(args: argparse.Namespace) -> None:
    for name in args.model:
        if server_running():
            call("DELETE", "/api/delete", {"model": name})
        else:
            aliases = config.load().get("aliases", {})
            if name in aliases:
                aliases.pop(name)
                config.update(aliases=aliases)
            else:
                r = store.resolve(lookup(name), online=False)
                store.delete(r.repo_id)
        print(f"deleted '{name}'")


def cmd_cp(args: argparse.Namespace) -> None:
    aliases = config.load().get("aliases", {})
    aliases[args.destination] = lookup(args.source)
    config.update(aliases=aliases)
    print(f"copied '{args.source}' to '{args.destination}'")


def cmd_ps(args: argparse.Namespace) -> None:
    need_server()
    rows = call("GET", "/api/ps")["models"]
    if not rows:
        print("no models loaded")
        return
    width = max(len(m["name"]) for m in rows)
    print(f"{'NAME':<{width}}  {'FAMILY':<16} {'PROCESSOR':<10} UNTIL")
    for m in rows:
        until = m["expires_at"][11:19] if m["expires_at"] else "forever"
        print(f"{m['name']:<{width}}  {m['details']['family']:<16} {m['device']:<10} {until}")


def cmd_stop(args: argparse.Namespace) -> None:
    need_server()
    print(call("POST", "/api/stop", {"model": args.model})["status"])


# ---- serve ------------------------------------------------------------------------------------------


def bind(host: str, port: int, scan: bool, tries: int = 50) -> tuple[socket.socket, int]:
    """Claim the port before the model loads, so a busy port fails in the first second."""
    candidates = range(port, port + tries) if scan else [port]
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    for candidate in candidates:
        sock = socket.socket(family)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, candidate))
        except OSError:
            sock.close()
            continue
        sock.listen(128)
        sock.set_inheritable(True)
        return sock, candidate
    if not scan:
        raise SystemExit(f"port {port} is already in use on {host}; pick another with --port")
    raise SystemExit(f"no free port in {port}..{port + tries - 1}")


def configure_logging(path: str) -> None:
    """Everything at INFO to the file, warnings and worse to the console."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    file = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.handlers = [file, console]


def url_host(host: str) -> str:
    probe = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"[{probe}]" if ":" in probe else probe


def cmd_serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    from . import api

    if getattr(args, "setup", False) or ("default_model" not in config.load() and sys.stdin.isatty() and not args.model):
        from .tui import setup

        if not setup():
            return
    data = config.load()
    env_host, env_port = config.host()
    host = args.host or data.get("host") or env_host
    explicit = args.port is not None or bool(os.environ.get("OLLAJEV_HOST"))
    port = args.port or (env_port if os.environ.get("OLLAJEV_HOST") else data.get("port") or env_port)
    model = args.model or default_model()

    configure_logging(args.log_file)
    sock, port = bind(host, port, scan=not explicit)
    base = f"http://{url_host(host)}:{port}"
    config.update(server_url=base)

    try:
        r = store.resolve(lookup(model), online=False)
        if store.is_trusted(r):
            api.preload = canonical(r)
        else:
            print(f"==> {canonical(r)} runs repo code and is not trusted yet; run: olla-jev pull {canonical(r)}")
    except LookupError:
        print(f"==> {model} is not downloaded; serving without a model. Pull one with: olla-jev pull {model}")
    if api.preload:
        print(f"==> Loading {api.preload}", flush=True)
        api.pin_preload = True
    threading.Thread(target=_announce_when_ready, args=(base, api.preload, not args.no_browser, args.log_file), daemon=True).start()
    uvicorn.Server(uvicorn.Config(api.app, log_config=None, log_level="info")).run(sockets=[sock])


def banner(base: str, model: str | None, log_file: str) -> str:
    return "\n".join([
        "",
        f"==> Ready on {base}" + (f", serving {model}" if model else ""),
        "",
        "    Jev / System One API",
        f"      GET   {base}/v1/models        downloaded models",
        f"      POST  {base}/v1/systemone     answer questions about a state (\"model\" picks the model)",
        "",
        "    Model management (Ollama style)",
        f"      GET   {base}/api/tags  ·  /api/ps  ·  POST /api/pull  ·  /api/show  ·  DELETE /api/delete",
        "",
        f"    Demo page   {base}/demo",
        "",
        "    For the TypeSafe SDK:",
        f"      export TYPESAFE_BASE_URL={base}",
        "      export TYPESAFE_API_KEY=local",
        "",
        f"    Logging to {log_file}. Ctrl-C to stop.",
        "",
    ])


def _announce_when_ready(base: str, model: str | None, open_browser: bool, log_file: str) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{base}/", timeout=1).close()
            break
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    else:
        return
    print(banner(base, model, log_file), flush=True)
    if open_browser:
        webbrowser.open(f"{base}/demo")


def cmd_setup(args: argparse.Namespace) -> None:
    args.setup = True
    cmd_serve(args)


def cmd_run(args: argparse.Namespace) -> None:
    from .run import run

    run(args.model)


# ---- parser -----------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="olla-jev", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=version("olla-jev"))
    parser.set_defaults(func=cmd_serve, model=None)

    def serve_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--host", help="bind address (default: 127.0.0.1, or OLLAJEV_HOST)")
        p.add_argument("--port", type=int, help="port (default: the first free one from 8000)")
        p.add_argument("--no-browser", action="store_true", help="do not open the demo page")
        p.add_argument("--log-file", default=DEFAULT_LOG, help=f"request and error log (default: {DEFAULT_LOG})")

    serve_options(parser)
    sub = parser.add_subparsers(dest="command", metavar="command")

    p = sub.add_parser("serve", help="start the server")
    p.add_argument("model", nargs="?", help="model to load at start (default: the saved default)")
    serve_options(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("setup", help="pick the default model and server options, then serve")
    serve_options(p)
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("run", help="ask a model questions interactively")
    p.add_argument("model", nargs="?", help="model (default: the saved default)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pull", help="download a model")
    p.add_argument("model", nargs="+", help="e.g. SupersonicLabs/Julia-1 or Mapika/decider-4b-GGUF:Q4_K_M")
    p.add_argument("--trust", action="store_true", help="trust the repo's Python code without asking")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("list", aliases=["ls"], help="list downloaded models")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("ps", help="list loaded models")
    p.set_defaults(func=cmd_ps)

    p = sub.add_parser("show", help="show a model's details and limits")
    p.add_argument("model")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("rm", help="delete a downloaded model")
    p.add_argument("model", nargs="+")
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("stop", help="unload a running model")
    p.add_argument("model")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("cp", help="copy a model under a new name")
    p.add_argument("source")
    p.add_argument("destination")
    p.set_defaults(func=cmd_cp)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (KeyboardInterrupt, EOFError):
        print()
    except (LookupError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None
