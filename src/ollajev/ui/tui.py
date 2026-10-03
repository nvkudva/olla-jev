"""Model manager: download, switch, ask, unload and delete models, and start the server, on one screen."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Select, Static, TextArea

from .. import client, config, service, store
from ..catalog import CATALOG
from ..manager import canonical, canonical_or, default_model, lookup
from ..server import admin
from . import repl

log = logging.getLogger(__name__)

CSS = """
Screen { background: $surface; }
Header { background: $surface; color: $text; }
DataTable { height: 1fr; background: $surface; padding: 0 1; }
DataTable > .datatable--header { background: $surface; color: $text-muted; text-style: bold; }
#status { height: 1; padding: 0 2; color: $text-muted; }
Footer { background: $surface; }

ModalScreen { align: center middle; background: $background 60%; }
.dialog { width: 68; height: auto; padding: 1 2; background: $panel; border: round $primary; }
.title { text-style: bold; margin-bottom: 1; }
.hint { color: $text-muted; margin-top: 1; }
.dialog Input, .dialog Select { margin-bottom: 1; }
.dialog TextArea { height: 6; margin-bottom: 1; }
#answers { height: auto; max-height: 12; }
.wide { width: 112; }
#results { height: 20; }
"""


class Prompt(ModalScreen[str | None]):
    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str, placeholder: str = "") -> None:
        super().__init__()
        self.heading, self.placeholder = title, placeholder

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(self.heading, classes="title")
            yield Input(placeholder=self.placeholder)
            yield Static("enter confirm · esc cancel", classes="hint")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


def human(size: float, units: tuple[str, ...] = ("B", "KB", "MB", "GB", "TB")) -> str:
    for unit in units[:-1]:
        if size < 1000:
            return f"{size:.0f} {unit}" if unit == units[0] else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} {units[-1]}"


class AddModel(ModalScreen[str | None]):
    """Search Hugging Face and list every quant of every matching repo in one table. Returns the name `pull`
    takes."""

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self.supported: set[str] = set()

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog wide"):
            yield Static("Add a model from Hugging Face", classes="title")
            yield Input(placeholder="search words, user/repo or a huggingface.co link", id="query")
            yield DataTable(id="results", cursor_type="row")
            yield Static("", id="note")
            yield Static("type to search · enter pick · esc cancel", classes="hint")

    def on_mount(self) -> None:
        table = self.query_one("#results", DataTable)
        table.add_column("Model", width=64)
        table.add_column("Size", width=9)
        table.add_column("Downloads", width=10)
        table.add_column("Support")

    def note(self, text: str) -> None:
        self.query_one("#note", Static).update(text)

    def on_input_changed(self, event: Input.Changed) -> None:
        self.search(event.value.strip(), delay=0.4)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.search(event.value.strip(), delay=0)
        self.query_one("#results", DataTable).focus()

    @work(exclusive=True, group="search")
    async def search(self, query: str, delay: float) -> None:
        await asyncio.sleep(delay)
        if not query:
            return
        self.note("Searching…")
        try:
            hits = await asyncio.to_thread(store.search, query, 25)
            self.note(f"Reading the quants of {len(hits)} models…")
            found = await asyncio.gather(*(asyncio.to_thread(variants, h.repo_id) for h in hits))
        except Exception as exc:
            log.exception("search failed")
            self.note(f"error: {exc}")
            return
        table = self.query_one("#results", DataTable)
        table.clear()
        self.supported = set()
        for h, vs in zip(hits, found, strict=True):
            support = f"✓ {h.family}" if h.family else "✗ unsupported"
            downloads = human(h.downloads, ("", "k", "M", "B")).replace(" ", "")
            for v in vs:
                table.add_row(v.name, human(v.size), downloads, support, key=v.name)
                if h.family:
                    self.supported.add(v.name)
        self.note("" if hits else "No models found")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()  # the manager behind this dialog downloads on its own row selection
        if event.row_key.value not in self.supported:
            self.notify("ollajev has no adapter for this model", severity="warning")
            return
        self.dismiss(event.row_key.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


def variants(repo_id: str) -> list[store.Variant]:
    """A repo's quants, or none when Hugging Face cannot list them; one failing repo must not fail a search."""
    try:
        return store.variants(repo_id)
    except Exception:
        log.exception("could not read %s", repo_id)
        return []


class Confirm(ModalScreen[bool]):
    BINDINGS: ClassVar = [("y", "yes", "Yes"), ("n,escape", "no", "No")]

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self.heading, self.body = title, body

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(self.heading, classes="title")
            yield Static(self.body)
            yield Static("y yes · n no", classes="hint")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class Info(ModalScreen[None]):
    BINDINGS: ClassVar = [("escape,enter,q", "close", "Close")]

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self.heading, self.body = title, body

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(self.heading, classes="title")
            yield Static(self.body)
            yield Static("esc close", classes="hint")

    def action_close(self) -> None:
        self.dismiss(None)


class Options(ModalScreen[dict[str, Any] | None]):
    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        saved = config.load()
        with Vertical(classes="dialog"):
            yield Static("Server options", classes="title")
            yield Static("Device")
            yield Select(
                [(d, d) for d in ("auto", "mps", "cuda", "cpu")],
                value=saved.get("device", "auto"),
                allow_blank=False,
                id="device",
            )
            yield Static("Address")
            yield Input(saved.get("host", "127.0.0.1"), id="host")
            yield Static("Port")
            yield Input(str(saved.get("port", config.DEFAULT_PORT)), id="port", type="integer")
            yield Static("tab next · enter save · esc cancel", classes="hint")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        port = self.query_one("#port", Input).value.strip()
        if not port.isdigit() or not 0 < int(port) < 65536:
            self.notify("Port must be 1-65535", severity="error")
            return
        self.dismiss(
            {
                "device": self.query_one("#device", Select).value,
                "host": self.query_one("#host", Input).value.strip() or "127.0.0.1",
                "port": int(port),
            }
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class Ask(ModalScreen[None]):
    """Ask a model questions. One question per line, in the same form `ollajev run` takes."""

    BINDINGS: ClassVar = [("escape", "close", "Close"), Binding("ctrl+r", "send", "Ask", priority=True)]

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model
        self.ask: Any = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(f"Ask {self.model}", classes="title")
            yield Static("State")
            yield TextArea(id="state")
            yield Static("Questions, one per line")
            yield TextArea(
                "noul: The customer asks for a refund.\nchoice: Which team? | billing, support, sales",
                id="questions",
            )
            yield Static("", id="answers")
            yield Static("ctrl+r ask · esc close", classes="hint")

    def on_mount(self) -> None:
        self.query_one("#answers", Static).update("Loading the model…")
        self.connect()

    @work(thread=True)
    def connect(self) -> None:
        try:
            self.ask = self.app.connection(self.model)  # type: ignore[attr-defined]
            self.app.call_from_thread(self.query_one("#answers", Static).update, "Ready. Press ctrl+r to ask.")
        except Exception as exc:
            log.exception("could not load %s", self.model)
            self.app.call_from_thread(self.query_one("#answers", Static).update, f"error: {exc}")

    def action_send(self) -> None:
        if self.ask is None:
            return
        state = self.query_one("#state", TextArea).text.strip()
        questions: dict[str, Any] = {}
        for line in self.query_one("#questions", TextArea).text.splitlines():
            if line.strip():
                parsed = repl.parse_question(line.strip())
                if parsed is None:
                    self.query_one("#answers", Static).update(f"not a question: {line.strip()}")
                    return
                questions[f"q{len(questions) + 1}"] = parsed[1]
        if not state or not questions:
            self.query_one("#answers", Static).update("Enter a state and at least one question.")
            return
        self.query_one("#answers", Static).update("Thinking…")
        self.send(state, questions)

    @work(thread=True, exclusive=True)
    def send(self, state: str, questions: dict[str, Any]) -> None:
        try:
            text = "\n".join(repl.format_answers(self.ask(state, questions)["answers"]))
        except Exception as exc:
            log.exception("ask failed")
            text = f"error: {exc}"
        self.app.call_from_thread(self.query_one("#answers", Static).update, text)

    def action_close(self) -> None:
        self.dismiss(None)


class Models(App[bool]):
    TITLE = "ollajev"
    CSS = CSS
    BINDINGS: ClassVar = [
        Binding("p", "pull", "Pull"),
        Binding("r", "ask", "Ask"),
        Binding("u", "unload", "Unload"),
        Binding("x", "remove", "Remove"),
        Binding("a", "alias", "Alias"),
        Binding("i", "info", "Info"),
        Binding("n", "add", "Add"),
        Binding("o", "options", "Options"),
        Binding("b", "service", "Service"),
        Binding("s", "serve", "Serve"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []
        self.busy = False
        self.local: tuple[str, Any, Any] | None = None  # model, ask, release: loaded in this process
        self.quants: dict[str, list[store.Variant]] = {}  # catalog GGUF repo -> all its quants on Hugging Face

    # ---- layout and data --------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(icon="")
        yield DataTable(cursor_type="row")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column("", width=2)
        table.add_column("Model", width=46)
        table.add_column("Size", width=8)
        table.add_column("Languages", width=16)
        table.add_column("Status")
        self.reload()
        self.say("Enter downloads a model and makes it the default. Press s to serve it.")
        self.load_quants()

    @work(group="quants")
    async def load_quants(self) -> None:
        """List every quant of the catalog's GGUF repos, not only the curated ones. Offline, the list stays as is."""
        repos = list(dict.fromkeys(e.name.partition(":")[0] for e in CATALOG if ":" in e.name))
        found = await asyncio.gather(*(asyncio.to_thread(variants, repo) for repo in repos))
        self.quants = {repo: vs for repo, vs in zip(repos, found, strict=True) if vs}
        self.reload()

    def say(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def loaded(self) -> set[str]:
        if client.server_running():
            try:
                return {m["name"] for m in client.call("GET", "/api/ps")["models"]}
            except SystemExit:
                return set()
        return {self.local[0]} if self.local else set()

    def reload(self) -> None:
        have = {m["name"]: m for m in admin.tags()}
        default = canonical_or(default_model())
        loaded = self.loaded()
        rows = []
        for i, e in enumerate(CATALOG):
            rows.append((e.name, e.name in have, f"{e.size_gb:.1f} GB", e.languages))
            repo = e.name.partition(":")[0]
            if repo in self.quants and all(n.name.partition(":")[0] != repo for n in CATALOG[i + 1 :]):
                curated = {n.name for n in CATALOG}
                rows += [
                    (v.name, v.name in have, human(v.size), e.languages)
                    for v in self.quants[repo]
                    if v.name not in curated
                ]
        listed = {r[0] for r in rows}
        rows += [
            (name, True, f"{m['size'] / 1e9:.1f} GB", m["details"]["family"])
            for name, m in have.items()
            if name not in listed
        ]
        table = self.query_one(DataTable)
        keep = self.names[table.cursor_row] if self.names and table.row_count else None
        table.clear()
        self.names = []
        for name, downloaded, size, languages in rows:
            status = " · ".join(s for s, on in (("default", name == default), ("loaded", name in loaded)) if on)
            table.add_row("✓" if downloaded else "", name, size, languages, status, key=name)
            self.names.append(name)
        if keep in self.names:
            table.move_cursor(row=self.names.index(keep))
        running = client.server_running()
        self.sub_title = f"server running at {client.server_url()}" if running else "server not running"

    def selected(self) -> str | None:
        table = self.query_one(DataTable)
        return self.names[table.cursor_row] if self.names else None

    def downloaded(self, name: str) -> bool:
        try:
            store.resolve(lookup(name), online=False)
        except (LookupError, ValueError):
            return False
        return True

    def connection(self, model: str) -> Any:
        """The ask function for `model`, loading it in this process unless a server is running."""
        if self.local and self.local[0] != model:
            self.release()
        if self.local is None:
            ask, release = repl.connect(model, lambda text: None)
            self.local = (model, ask, release)
        return self.local[1]

    def release(self) -> None:
        if self.local:
            self.local[2]()
            self.local = None

    # ---- jobs: one at a time, off the UI thread ---------------------------------------------------

    async def job(self, text: str, work_fn: Any) -> Any:
        if self.busy:
            self.notify("Wait for the current job to finish", severity="warning")
            return None
        self.busy = True
        self.say(text)
        try:
            return await work_fn()
        except Exception as exc:
            log.exception("%s failed", text)
            self.notify(str(exc) or type(exc).__name__, severity="error", timeout=10)
            return None
        finally:
            self.busy = False
            self.reload()
            self.say("")

    async def fetch(self, name: str) -> store.Resolved | None:
        r = await asyncio.to_thread(lambda: store.resolve(lookup(name)))
        if r.family.runs_repo_code and not store.is_trusted(r):
            code = sorted(f for f in r.files if f.endswith(".py"))
            body = (
                f"{canonical(r)} runs Python code from its Hugging Face repo, with your user's privileges.\n\n"
                f"commit  {r.revision}\nreview  https://huggingface.co/{r.repo_id}/tree/{r.revision}\n"
                f"files   {', '.join(code[:6])}{' …' if len(code) > 6 else ''}\n\nTrust this exact commit?"
            )
            if not await self.push_screen_wait(Confirm("This model runs repo code", body)):
                self.notify("Not trusted; nothing downloaded", severity="warning")
                return None
            store.trust(r)
        self.say(f"Downloading {canonical(r)} …")
        await asyncio.to_thread(store.download, r)
        prefetch = getattr(r.family, "prefetch", None)
        if prefetch:
            self.say("Downloading the base model …")
            await asyncio.to_thread(prefetch, store.local_path(r))
        return r

    # ---- actions ----------------------------------------------------------------------------------

    @work
    async def on_data_table_row_selected(self) -> None:
        """Enter: download if needed, then make it the default."""
        name = self.selected()
        if not name:
            return

        async def use() -> None:
            if r := await self.fetch(name):
                config.update(default_model=canonical(r))
                self.notify(f"{canonical(r)} is now the default")

        await self.job(f"Preparing {name} …", use)

    @work
    async def action_pull(self) -> None:
        if name := self.selected():

            async def pull() -> None:
                if r := await self.fetch(name):
                    self.notify(f"Downloaded {canonical(r)}")

            await self.job(f"Preparing {name} …", pull)

    @work
    async def action_add(self) -> None:
        name = await self.push_screen_wait(AddModel())
        if name:

            async def pull() -> None:
                if r := await self.fetch(name):
                    self.notify(f"Downloaded {canonical(r)}")

            await self.job(f"Preparing {name} …", pull)

    @work
    async def action_ask(self) -> None:
        name = self.selected()
        if not name:
            return
        if not self.downloaded(name):
            self.notify("Download the model first (Enter)", severity="warning")
            return
        await self.push_screen_wait(Ask(name))
        self.reload()

    @work
    async def action_unload(self) -> None:
        name = self.selected()
        if not name:
            return
        if client.server_running():
            self.notify(await asyncio.to_thread(lambda: client.call("POST", "/api/stop", {"model": name})["status"]))
        elif self.local and self.local[0] == name:
            await asyncio.to_thread(self.release)
            self.notify("unloaded")
        else:
            self.notify("not loaded")
        self.reload()

    @work
    async def action_remove(self) -> None:
        name = self.selected()
        if not name or not self.downloaded(name):
            return
        if not await self.push_screen_wait(Confirm(f"Delete {name}?", "The downloaded weights are removed from disk.")):
            return

        async def remove() -> None:
            if self.local and self.local[0] == name:
                await asyncio.to_thread(self.release)
            if client.server_running():
                await asyncio.to_thread(lambda: client.call("DELETE", "/api/delete", {"model": name}))
            else:
                await asyncio.to_thread(lambda: store.remove(store.resolve(lookup(name), online=False)))
            self.notify(f"Deleted {name}")

        await self.job(f"Deleting {name} …", remove)

    @work
    async def action_alias(self) -> None:
        name = self.selected()
        if not name:
            return
        short = await self.push_screen_wait(Prompt(f"Short name for {name}", "julia"))
        if short:
            with config.edit() as data:
                data.setdefault("aliases", {})[short] = name
            self.notify(f"'{short}' now means {name}")

    @work
    async def action_info(self) -> None:
        name = self.selected()
        if not name:
            return
        try:
            r = store.resolve(lookup(name), online=False)
        except (LookupError, ValueError):
            entry = next((e for e in CATALOG if e.name == name), None)
            await self.push_screen_wait(
                Info(
                    name, f"{entry.description if entry else 'unknown model'}\n\nNot downloaded. Press Enter to get it."
                )
            )
            return
        trusted = (
            "no repo code" if not r.family.runs_repo_code else ("trusted" if store.is_trusted(r) else "NOT trusted")
        )
        limits = ", ".join(f"{k} {v}" for k, v in r.family.limits(r).items()) or "none recorded"
        body = (
            f"family   {r.family.name}\ncommit   {r.revision}\nfile     {r.gguf or 'safetensors'}\n"
            f"code     {trusted}\nlimits   {limits}\npath     {store.local_path(r)}"
        )
        await self.push_screen_wait(Info(canonical(r), body))

    @work
    async def action_options(self) -> None:
        if values := await self.push_screen_wait(Options()):
            config.update(**values)
            self.notify("Saved. Options apply the next time the server starts.")

    @work
    async def action_service(self) -> None:
        try:
            running, _ = await asyncio.to_thread(service.status)
            if running:
                if await self.push_screen_wait(Confirm("Background service", "Stop and remove it?")):
                    self.notify(await asyncio.to_thread(service.uninstall))
            elif await self.push_screen_wait(Confirm("Background service", "Run ollajev in the background at login?")):
                self.notify(await asyncio.to_thread(service.install))
        except SystemExit as exc:
            self.notify(str(exc), severity="error")

    def action_serve(self) -> None:
        self.release()
        self.exit(True)

    def action_quit_app(self) -> None:
        self.release()
        self.exit(False)


def manage() -> bool:
    """Open the model manager. True when the user chose to serve."""
    return bool(Models().run())
