"""Model manager: download, switch, ask, unload and delete models, and start the server, on one screen."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, NamedTuple

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, DataTable, Footer, Header, Static

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
#actions { height: 1; padding: 0 1; margin-top: 1; }
.buttons { height: auto; margin-top: 1; }
.buttons Button { width: auto; min-width: 0; padding: 0 1; margin-right: 1; background: $boost; }
.buttons Button:hover { background: $primary 40%; }
.buttons Button.-primary { background: $primary; }
.buttons Button.-success { background: $success 70%; }
.buttons Button.-error { background: $error 70%; }
.buttons .hint { width: 1fr; margin-top: 0; }
DataTable > .datatable--header { background: $surface; color: $text-muted; text-style: bold; }
#status { height: 1; padding: 0 2; margin-bottom: 1; color: $text-muted; }
Footer { background: $surface; }

ModalScreen { align: center middle; background: $background 60%; }
.dialog { width: 68; max-width: 96%; height: auto; padding: 1 2; background: $panel; border: round $primary; }
.title { text-style: bold; margin-bottom: 1; }
.hint { color: $text-muted; margin-top: 1; }
.dialog Input, .dialog Select { margin-bottom: 1; }
.dialog TextArea { height: 6; margin-bottom: 1; }
#answers-box { height: auto; max-height: 14; }
.wide { width: 112; }
#results { height: 20; }
"""


class NoWaitExecutor(ThreadPoolExecutor):
    """The event loop's default executor. On exit asyncio waits for every thread in it to finish, and a Hugging Face
    lookup still running would keep the terminal blank after `s` or `q`. This one lets them finish on their own."""

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        super().shutdown(wait=False, cancel_futures=True)


class Row(NamedTuple):
    name: str  # what pull, rm and the config call it
    label: str  # what the Model column shows
    size: str
    languages: str  # the catalog's languages, or the family of a model outside the catalog


class Models(App[bool]):
    TITLE = "ollajev"
    CSS = CSS
    # The footer shows the everyday keys; ? lists them all, so the footer fits a narrow terminal.
    BINDINGS: ClassVar = [
        Binding("p", "pull", "Pull"),
        Binding("r", "ask", "Ask"),
        Binding("n", "add", "Add"),
        Binding("x", "remove", "Remove"),
        Binding("s", "serve", "Serve"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit_app", "Quit"),
        Binding("d", "set_default", "Default", show=False),
        Binding("u", "unload", "Unload", show=False),
        Binding("a", "alias", "Alias", show=False),
        Binding("i", "info", "Info", show=False),
        Binding("o", "options", "Options", show=False),
        Binding("b", "service", "Service", show=False),
        Binding("slash", "filter", "Filter", show=False),
        Binding("ctrl+r", "reload_list", "Refresh", show=False),
        Binding("e", "last_error", "Error", show=False),
        Binding("escape", "cancel_job", "Cancel", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []
        self.sizes: dict[str, str] = {}  # model name -> the size shown in its row
        self.summary = ""  # the idle status line: default, loaded, disk use, server
        self.filter_text = ""  # `/` shows only the models whose name contains it
        self.busy = False
        self.cancel = threading.Event()  # set by Esc; a download in progress stops at its next update
        self.downloading = False  # only downloads can be cancelled; loads and asks run to the end
        self.local: tuple[str, Any, Any] | None = None  # model, ask, release: loaded in this process
        self.last_error: str | None = None  # kept on the status line until the next job succeeds
        self.loading: str | None = None  # the model being loaded into this process, if any
        # One load at a time: two Ask dialogs opened in a row must not hold two models in memory.
        self.load_lock = threading.Lock()
        self.quants: dict[str, list[store.Variant]] = {}  # catalog GGUF repo -> all its quants on Hugging Face

    # ---- layout and data --------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(icon="")
        yield DataTable(cursor_type="row", zebra_stripes=True)
        yield dialogs.buttons(
            ("Download", "pull", "primary"),
            ("Ask", "ask", "default"),
            ("Make default", "set_default", "default"),
            ("Info", "info", "default"),
            ("Unload", "unload", "default"),
            ("Delete", "remove", "default"),
            ("Add model…", "add", "default"),
            ("Serve", "serve", "success"),
            row_id="actions",
        )
        yield Static("", id="status")
        yield Footer()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """A button acts on the selected row, like its key; then the list takes the keys again."""
        await dialogs.Clickable.on_button_pressed(self, event)  # type: ignore[arg-type]
        self.query_one(DataTable).focus()

    def on_mount(self) -> None:
        asyncio.get_running_loop().set_default_executor(NoWaitExecutor())
        table = self.query_one(DataTable)
        table.add_column("Disk", width=4)
        table.add_column("Model", width=46)
        table.add_column("Size", width=9)
        table.add_column("Languages / family", width=18)
        table.add_column("Status")
        self.reload()
        self.say_idle()
        self.load_quants()
        self.set_interval(3, self.auto_refresh)

    @work(group="quants")
    async def load_quants(self) -> None:
        """List every quant of the catalog's GGUF repos, not only the curated ones. Offline, the list stays as is."""
        repos = list(dict.fromkeys(e.name.partition(":")[0] for e in CATALOG if ":" in e.name))
        found = await asyncio.gather(*(asyncio.to_thread(dialogs.variants, repo) for repo in repos))
        self.quants = {repo: vs for repo, vs in zip(repos, found, strict=True) if vs}
        self.reload()

    def say(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def say_idle(self) -> None:
        """The status line between jobs: the last error until a job succeeds, else the main hint."""
        if self.last_error:
            first_line = self.last_error.splitlines()[0]
            self.say(f"! {first_line}  (e for details)")
        else:
            self.say(self.summary)

    def action_help(self) -> None:
        self.push_screen(dialogs.Info("Keys", KEYS_HELP))

    def action_last_error(self) -> None:
        if self.last_error:
            self.push_screen(dialogs.Info("Last error", self.last_error))
        else:
            self.notify("No errors so far")

    def loaded(self, server_up: bool) -> set[str]:
        if server_up:
            try:
                return {m["name"] for m in client.call("GET", "/api/ps")["models"]}
            except SystemExit:
                return set()
        return {self.local[0]} if self.local else set()

    def snapshot(self) -> tuple[dict[str, Any], str, bool, set[str]]:
        """What the list shows: downloads, default, server up, loaded models. Slow: disk scan and server probes."""
        have = {m["name"]: m for m in admin.tags()}
        default = canonical_or(default_model())
        server_up = client.server_running()  # each probe can wait 0.5 s, so probe once per reload
        return have, default, server_up, self.loaded(server_up)

    def reload(self) -> None:
        self.render_list(*self.snapshot())

    @work(thread=True, exclusive=True, group="auto-refresh")
    def auto_refresh(self) -> None:
        """Every 3 s: pick up a server, download or default changed elsewhere, off the UI thread."""
        if self.busy or self.loading or len(self.screen_stack) > 1:
            return
        data = self.snapshot()
        self.call_from_thread(self.render_list, *data)

    def render_list(self, have: dict[str, Any], default: str, server_up: bool, loaded: set[str]) -> None:
        rows = self.list_rows(have)
        if self.filter_text:
            rows = [row for row in rows if self.filter_text.lower() in row.name.lower()]
        table = self.query_one(DataTable)
        keep = self.names[table.cursor_row] if self.names and table.row_count else None
        table.clear()
        self.names = []
        self.sizes = {}
        for row in rows:
            self.sizes[row.name] = row.size
            markers = Text()
            if row.name == default:
                markers.append("★ default  ", style="bold yellow")
            if row.name in loaded:
                markers.append("● loaded", style="bold green")
            disk = Text("✓", style="green") if row.name in have else ""
            table.add_row(disk, row.label, row.size, row.languages, markers, key=row.name)
            self.names.append(row.name)
        if keep in self.names:
            table.move_cursor(row=self.names.index(keep))
        self.sub_title = f"server running at {client.server_url()}" if server_up else "server not running"
        self.summary = self.describe(default, loaded, have, server_up)
        if not self.busy:
            self.say_idle()

    def list_rows(self, have: dict[str, Any]) -> list[Row]:
        """The catalog, each GGUF repo's other quants indented under its last curated entry, then any other
        download. Sizes on disk are exact; a size not yet downloaded is an estimate, marked ~."""
        curated = {entry.name for entry in CATALOG}
        rows = []
        for index, entry in enumerate(CATALOG):
            rows.append(
                Row(entry.name, entry.name, self.size_of(entry.name, have, entry.size_gb * 1e9), entry.languages)
            )
            repo = entry.name.partition(":")[0]
            later_entries = CATALOG[index + 1 :]
            last_of_repo = all(other.name.partition(":")[0] != repo for other in later_entries)
            if not last_of_repo:
                continue
            for variant in self.quants.get(repo, []):
                if variant.name in curated:
                    continue
                label = f"  └ {variant.name.partition(':')[2] or variant.name}"
                rows.append(Row(variant.name, label, self.size_of(variant.name, have, variant.size), entry.languages))
        listed = {row.name for row in rows}
        for name, model in have.items():
            if name not in listed:
                rows.append(Row(name, name, dialogs.human(model["size"]), model["details"]["family"]))
        return rows

    @staticmethod
    def size_of(name: str, have: dict[str, Any], estimate: float) -> str:
        if name in have:
            return dialogs.human(have[name]["size"])
        return "~" + dialogs.human(estimate)

    def describe(self, default: str, loaded: set[str], have: dict[str, Any], server_up: bool) -> str:
        on_disk = sum(model["size"] for model in have.values())
        server = client.server_url() if server_up else "not running"
        parts = [
            f"default {default}",
            f"loaded {', '.join(sorted(loaded)) or 'none'}",
            f"{len(have)} on disk ({dialogs.human(on_disk)})",
            f"server {server}",
        ]
        if self.filter_text:
            parts.append(f"filter '{self.filter_text}'")
        parts.append("? keys")
        return " · ".join(parts)

    @work
    async def action_filter(self) -> None:
        """`/`: show only matching models; an empty filter shows them all again."""
        text = await self.push_screen_wait(dialogs.Prompt("Filter models", "part of a name; empty shows all"))
        self.filter_text = text or ""
        self.reload()

    def action_reload_list(self) -> None:
        self.reload()
        self.notify("Refreshed")

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
            result = await work_fn()
            self.last_error = None
            return result
        except store.Cancelled:
            self.notify("Cancelled; a later pull resumes where it stopped", severity="warning")
            return None
        except Exception as exc:
            log.exception("%s failed", text)
            self.last_error = f"{text.rstrip(' …')} failed:\n{exc or type(exc).__name__}"
            self.notify(str(exc) or type(exc).__name__, severity="error", timeout=10)
            return None
        finally:
            self.busy = False
            self.reload()
            self.say_idle()

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
        self.downloading = True
        progress = await self.show_progress(resolved)
        try:
            await asyncio.to_thread(store.download, resolved, self.cancel)
            if store.needs_prefetch(resolved):
                self.say("Downloading the base model …")
                await asyncio.to_thread(store.prefetch, resolved, self.cancel)
        finally:
            progress.stop()
            self.downloading = False
        return resolved

    async def show_progress(self, resolved: store.Resolved) -> Any:
        """Put the download's bytes, percent and speed on the status line every half second; returns the timer."""
        name = canonical(resolved)
        try:
            total = await asyncio.to_thread(store.download_size, resolved)
        except Exception:  # progress is optional; the download itself reports real errors
            total = 0
        start = store.bytes_on_disk(resolved.repo_id)
        started = time.monotonic()

        def update() -> None:
            done = store.bytes_on_disk(resolved.repo_id) - start
            speed = done / max(time.monotonic() - started, 0.001)
            if total:
                percent = min(100, done * 100 // total)
                amount = f"{dialogs.human(done)} / {dialogs.human(total)} · {percent}%"
            else:
                amount = dialogs.human(done)
            self.say(f"Downloading {name}: {amount} · {dialogs.human(speed)}/s · esc cancels")

        update()
        return self.set_interval(0.5, update)

    # ---- actions ----------------------------------------------------------------------------------

    @work
    async def on_data_table_row_selected(self) -> None:
        """Enter: download if needed, then make it the default."""
        name = self.selected()
        if not name:
            return
        if not self.downloaded(name):
            size = self.sizes.get(name, "an unknown size")
            question = dialogs.Confirm(
                f"Download {name}?", f"It is {size}. It becomes the default model.", default=True
            )
            if not await self.push_screen_wait(question):
                return

        async def use() -> None:
            if resolved := await self.fetch(name):
                config.update(default_model=canonical(resolved))
                self.notify(f"{canonical(resolved)} is now the default")

        await self.job(f"Preparing {name} …", use)

    def action_set_default(self) -> None:
        name = self.selected()
        if not name:
            return
        if not self.downloaded(name):
            self.notify("Download it first (p or Enter)", severity="warning")
            return
        config.update(default_model=canonical_or(name))
        self.notify(f"{canonical_or(name)} is now the default")
        self.reload()

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
        if not short:
            return
        if any(char.isspace() for char in short) or short in self.names:
            self.notify(f"'{short}' cannot be a short name: no spaces, and not a model's own name", severity="error")
            return
        current = config.load().get("aliases", {}).get(short)
        if current and current != name:
            replace = dialogs.Confirm(f"Replace '{short}'?", f"It means {current} now.")
            if not await self.push_screen_wait(replace):
                return
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
                    name, f"{entry.description if entry else 'unknown model'}\n\nNot downloaded. Press Enter or Download to get it."
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
                dialogs.Confirm("Background service", "Run ollajev in the background at login?", default=True)
            ):
                self.notify(await asyncio.to_thread(service.install))
        except SystemExit as exc:
            self.notify(str(exc), severity="error")

    def action_serve(self) -> None:
        if self.refuse_while_busy():
            return
        if client.server_running():
            self.notify(f"A server is already running at {client.server_url()}", severity="warning")
            return
        self.release()
        self.exit(True)

    @work
    async def action_quit_app(self) -> None:
        if self.downloading:
            question = dialogs.Confirm("A download is running", "Quit anyway? The next pull resumes it.")
            if not await self.push_screen_wait(question):
                return
        # Stop a download at its next update. A loaded model is not unloaded: the process is about to end,
        # which frees it at once, while unloading it here would freeze the screen first.
        self.cancel.set()
        self.exit(False)

    async def action_quit(self) -> None:
        """ctrl+q: the same as q."""
        self.action_quit_app()


KEYS_HELP = """\
enter   download if needed, make it the default   (pull + default)
d       make a downloaded model the default
p       download only                              (pull)
r       ask the model questions                    (run)
n       add any Hugging Face repo by name          (pull)
x       delete the download                        (rm)
u       unload it from memory                      (stop)
a       give it a short name                       (cp)
i       family, commit, limits, path               (show)
/       filter the list by name
ctrl+r  refresh the list
o       device, address and port for the server
b       install or remove the background service   (service)
s       start the server and leave the manager     (serve)
e       the last error in full
esc     cancel a download
q       quit"""


def manage() -> bool:
    """Open the model manager. True when the user chose to serve."""
    return bool(Models().run())
