"""The model manager's dialogs: prompts, model search, confirmations, info, options and the ask screen."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
from typing import Any, ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Select, Static, TextArea

from .. import client, config, store
from . import repl

log = logging.getLogger(__name__)


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
        except Exception as exc:
            log.exception("search failed")
            self.note(f"error: {exc}")
            return
        if not hits:
            self.show_results(hits, {})
            self.note("No models found")
            return
        # Each repo's quants take a Hugging Face call; show the rows as they arrive, in search order.
        quants: dict[str, list[store.Variant]] = {}

        async def read(hit: store.Hit) -> None:
            quants[hit.repo_id] = await asyncio.to_thread(variants, hit.repo_id)
            self.show_results(hits, quants)
            self.note(f"Reading quants: {len(quants)}/{len(hits)} models")

        await asyncio.gather(*(read(hit) for hit in hits))
        self.note("")

    def show_results(self, hits: list[store.Hit], quants: dict[str, list[store.Variant]]) -> None:
        table = self.query_one("#results", DataTable)
        cursor = table.cursor_row  # rows keep arriving while the user moves through them
        table.clear()
        self.supported = set()
        for hit in hits:
            support = f"✓ {hit.family}" if hit.family else "✗ unsupported"
            downloads = human(hit.downloads, ("", "k", "M", "B")).replace(" ", "")
            for variant in quants.get(hit.repo_id, []):
                table.add_row(variant.name, human(variant.size), downloads, support, key=variant.name)
                if hit.family:
                    self.supported.add(variant.name)
        if table.row_count:
            table.move_cursor(row=min(cursor, table.row_count - 1))

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
    """Yes or no. Enter picks `default`: yes for harmless steps, no for anything that deletes or discards."""

    BINDINGS: ClassVar = [("y", "yes", "Yes"), ("n,escape", "no", "No"), ("enter", "default", "Default")]

    def __init__(self, title: str, body: str, default: bool = False) -> None:
        super().__init__()
        self.heading, self.body, self.default = title, body, default

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(self.heading, classes="title")
            yield Static(self.body)
            hint = "y yes · n no · enter yes" if self.default else "y yes · n no · enter no"
            yield Static(hint, classes="hint")

    def action_default(self) -> None:
        self.dismiss(self.default)

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


def valid_host(host: str) -> bool:
    """A host name or an IP address, with no port: a colon is only allowed inside an IPv6 address."""
    if not host or any(char.isspace() for char in host):
        return False
    if ":" not in host:
        return True
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return True


class Options(ModalScreen[dict[str, Any] | None]):
    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        saved = config.load()
        with Vertical(classes="dialog"):
            yield Static("Server options", classes="title")
            yield Static("Device (auto picks cuda, then mps, then cpu)")
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
        host = self.query_one("#host", Input).value.strip() or "127.0.0.1"
        if not valid_host(host):
            self.notify("Address must be a host name or IP address, without a port", severity="error")
            return
        if not config.is_loopback(host) and not config.api_key():
            self.notify(f"Serving on {host} needs OLLAJEV_API_KEY set first", severity="error")
            return
        self.dismiss({"device": self.query_one("#device", Select).value, "host": host, "port": int(port)})

    def action_cancel(self) -> None:
        self.dismiss(None)


class Ask(ModalScreen[None]):
    """Ask a model questions. One question per line, in the same form `ollajev run` takes."""

    BINDINGS: ClassVar = [("escape", "close", "Close"), Binding("ctrl+s,ctrl+r", "send", "Ask", priority=True)]

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model
        self.ask: Any = None
        self.history: list[str] = []  # earlier answers, newest first, kept while the dialog is open

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(f"Ask {self.model}", classes="title")
            yield Static("State")
            yield TextArea(id="state")
            yield Static("Questions, one per line")
            yield TextArea(
                id="questions",
                placeholder="noul: The customer asks for a refund.\nchoice: Which team? | billing, support, sales",
            )
            with VerticalScroll(id="answers-box"):
                yield Static("", id="answers")
            yield Static("ctrl+s ask · esc close", classes="hint")

    def on_mount(self) -> None:
        self.show(f"Loading {self.model} …")
        self.connect()

    def show(self, text: str) -> None:
        # A load or answer can finish after Esc closed this dialog; there is nothing left to update then.
        if self.is_attached:
            self.query_one("#answers", Static).update(text)

    @work(thread=True)
    def connect(self) -> None:
        where = "on the server" if client.server_running() else "into this process (no server running)"
        self.app.call_from_thread(self.show, f"Loading {self.model} {where} …")
        try:
            self.ask = self.app.connection(self.model)  # type: ignore[attr-defined]
            self.app.call_from_thread(self.show, "Ready. Press ctrl+s to ask.")
        except Exception as exc:
            log.exception("could not load %s", self.model)
            self.app.call_from_thread(self.show, f"error: {exc}")

    def action_send(self) -> None:
        if self.ask is None:
            self.show("Still loading the model…")
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
        started = time.monotonic()
        try:
            text = "\n".join(repl.format_answers(self.ask(state, questions)["answers"]))
        except Exception as exc:
            log.exception("ask failed")
            text = f"error: {exc}"
        seconds = time.monotonic() - started
        self.history.insert(0, f"── {self.model} · {seconds:.1f} s\n{text}")
        self.app.call_from_thread(self.show, "\n\n".join(self.history))

    def action_close(self) -> None:
        self.dismiss(None)
