"""Model manager: download, switch, ask, unload and delete models, and start the server, on one screen."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

from .. import client, config, service, store
from ..catalog import CATALOG
from ..manager import canonical, canonical_or, default_model, lookup
from ..server import admin
from . import dialogs, repl

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
        Binding("escape", "cancel_job", "Cancel"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []
        self.busy = False
        self.cancel = threading.Event()  # set by Esc; a download in progress stops at its next update
        self.downloading = False  # only downloads can be cancelled; loads and asks run to the end
        self.local: tuple[str, Any, Any] | None = None  # model, ask, release: loaded in this process
        self.loading: str | None = None  # the model being loaded into this process, if any
        # One load at a time: two Ask dialogs opened in a row must not hold two models in memory.
        self.load_lock = threading.Lock()
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
        found = await asyncio.gather(*(asyncio.to_thread(dialogs.variants, repo) for repo in repos))
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
                    (v.name, v.name in have, dialogs.human(v.size), e.languages)
                    for v in self.quants[repo]
                    if v.name not in curated
                ]
        listed = {row[0] for row in rows}
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
        with self.load_lock:
            if self.local and self.local[0] != model:
                self.release()
            if self.local is None:
                self.loading = model
                try:
                    ask, release = repl.connect(model, lambda text: None)
                finally:
                    self.loading = None
                self.local = (model, ask, release)
            return self.local[1]

    def refuse_while_busy(self) -> bool:
        """True, with a message, when a job or a model load is running and the action must wait."""
        if self.busy:
            self.notify("Wait for the current job to finish (esc cancels a download)", severity="warning")
            return True
        if self.loading:
            self.notify(f"Wait for {self.loading} to finish loading", severity="warning")
            return True
        return False

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
        self.cancel.clear()
        self.say(text)
        try:
            return await work_fn()
        except store.Cancelled:
            self.notify("Cancelled; a later pull resumes where it stopped", severity="warning")
            return None
        except Exception as exc:
            log.exception("%s failed", text)
            self.notify(str(exc) or type(exc).__name__, severity="error", timeout=10)
            return None
        finally:
            self.busy = False
            self.reload()
            self.say("")

    def action_cancel_job(self) -> None:
        if self.downloading:
            self.cancel.set()
            self.say("Cancelling …")
        elif self.busy:
            self.notify("Only downloads can be cancelled", severity="warning")

    async def fetch(self, name: str) -> store.Resolved | None:
        resolved = await asyncio.to_thread(lambda: store.resolve(lookup(name)))
        if resolved.family.runs_repo_code and not store.is_trusted(resolved):
            code = sorted(f for f in resolved.files if f.endswith(".py"))
            body = (
                f"{canonical(resolved)} runs Python code from its Hugging Face repo, with your user's privileges.\n\n"
                f"commit  {resolved.revision}\nreview  https://huggingface.co/{resolved.repo_id}/tree/{resolved.revision}\n"
                f"files   {', '.join(code[:6])}{' …' if len(code) > 6 else ''}\n\nTrust this exact commit?"
            )
            if not await self.push_screen_wait(dialogs.Confirm("This model runs repo code", body)):
                self.notify("Not trusted; nothing downloaded", severity="warning")
                return None
            store.trust(resolved)
        self.say(f"Downloading {canonical(resolved)} …")
        self.downloading = True
        try:
            await asyncio.to_thread(store.download, resolved, self.cancel)
            if store.needs_prefetch(resolved):
                self.say("Downloading the base model …")
                await asyncio.to_thread(store.prefetch, resolved, self.cancel)
        finally:
            self.downloading = False
        return resolved

    # ---- actions ----------------------------------------------------------------------------------

    @work
    async def on_data_table_row_selected(self) -> None:
        """Enter: download if needed, then make it the default."""
        name = self.selected()
        if not name:
            return

        async def use() -> None:
            if resolved := await self.fetch(name):
                config.update(default_model=canonical(resolved))
                self.notify(f"{canonical(resolved)} is now the default")

        await self.job(f"Preparing {name} …", use)

    @work
    async def action_pull(self) -> None:
        if name := self.selected():

            async def pull() -> None:
                if resolved := await self.fetch(name):
                    self.notify(f"Downloaded {canonical(resolved)}")

            await self.job(f"Preparing {name} …", pull)

    @work
    async def action_add(self) -> None:
        name = await self.push_screen_wait(dialogs.AddModel())
        if name:

            async def pull() -> None:
                if resolved := await self.fetch(name):
                    self.notify(f"Downloaded {canonical(resolved)}")

            await self.job(f"Preparing {name} …", pull)

    @work
    async def action_ask(self) -> None:
        name = self.selected()
        if not name:
            return
        if not self.downloaded(name):
            self.notify("Download the model first (Enter)", severity="warning")
            return
        await self.push_screen_wait(dialogs.Ask(name))
        self.reload()

    @work
    async def action_unload(self) -> None:
        name = self.selected()
        if not name or self.refuse_while_busy():
            return
        if client.server_running():
            reply = await asyncio.to_thread(client.call, "POST", "/api/stop", {"model": name})
            self.notify(reply["status"])
        elif self.local and self.local[0] == name:
            await asyncio.to_thread(self.release)
            self.notify(f"Unloaded {name}")
        else:
            self.notify(f"{name} is not loaded")
        self.reload()

    @work
    async def action_remove(self) -> None:
        name = self.selected()
        if not name or self.refuse_while_busy():
            return
        if not self.downloaded(name):
            self.notify(f"{name} is not downloaded; nothing to delete")
            return
        if not await self.push_screen_wait(
            dialogs.Confirm(f"Delete {name}?", "The downloaded weights are removed from disk.")
        ):
            return

        async def remove() -> None:
            if self.local and self.local[0] == name:
                await asyncio.to_thread(self.release)
            if client.server_running():
                await asyncio.to_thread(client.call, "DELETE", "/api/delete", {"model": name})
            else:
                resolved = await asyncio.to_thread(store.resolve, lookup(name), online=False)
                await asyncio.to_thread(store.remove, resolved)
            self.notify(f"Deleted {name}")

        await self.job(f"Deleting {name} …", remove)

    @work
    async def action_alias(self) -> None:
        name = self.selected()
        if not name:
            return
        short = await self.push_screen_wait(dialogs.Prompt(f"Short name for {name}", "julia"))
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
            resolved = store.resolve(lookup(name), online=False)
        except (LookupError, ValueError):
            entry = next((e for e in CATALOG if e.name == name), None)
            await self.push_screen_wait(
                dialogs.Info(
                    name, f"{entry.description if entry else 'unknown model'}\n\nNot downloaded. Press Enter to get it."
                )
            )
            return
        trusted = store.trust_label(resolved)
        limits = ", ".join(f"{k} {v}" for k, v in resolved.family.limits(resolved).items()) or "none recorded"
        body = (
            f"family   {resolved.family.name}\ncommit   {resolved.revision}\nfile     {resolved.weights or 'safetensors'}\n"
            f"code     {trusted}\nlimits   {limits}\npath     {store.local_path(resolved)}"
        )
        await self.push_screen_wait(dialogs.Info(canonical(resolved), body))

    @work
    async def action_options(self) -> None:
        if values := await self.push_screen_wait(dialogs.Options()):
            config.update(**values)
            self.notify("Saved. Options apply the next time the server starts.")

    @work
    async def action_service(self) -> None:
        if self.refuse_while_busy():
            return
        try:
            running, _ = await asyncio.to_thread(service.status)
            if running:
                if await self.push_screen_wait(dialogs.Confirm("Background service", "Stop and remove it?")):
                    self.notify(await asyncio.to_thread(service.uninstall))
            elif await self.push_screen_wait(
                dialogs.Confirm("Background service", "Run ollajev in the background at login?")
            ):
                self.notify(await asyncio.to_thread(service.install))
        except SystemExit as exc:
            self.notify(str(exc), severity="error")

    def action_serve(self) -> None:
        if self.refuse_while_busy():
            return
        self.release()
        self.exit(True)

    def action_quit_app(self) -> None:
        # Stop a download at its next update. A loaded model is not unloaded: the process is about to end,
        # which frees it at once, while unloading it here would freeze the screen first.
        self.cancel.set()
        self.exit(False)

    async def action_quit(self) -> None:
        """ctrl+q: the same as q."""
        self.action_quit_app()


def manage() -> bool:
    """Open the model manager. True when the user chose to serve."""
    return bool(Models().run())
