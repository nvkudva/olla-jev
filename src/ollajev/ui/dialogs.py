"""The model manager's dialogs: prompts, model search, confirmations, info, options and the ask screen."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Select, Static, TextArea

from .. import config, store
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
        self.show("Loading the model…")
        self.connect()

    def show(self, text: str) -> None:
        # A load or answer can finish after Esc closed this dialog; there is nothing left to update then.
        if self.is_attached:
            self.query_one("#answers", Static).update(text)

    @work(thread=True)
    def connect(self) -> None:
        try:
            self.ask = self.app.connection(self.model)  # type: ignore[attr-defined]
            self.app.call_from_thread(self.show, "Ready. Press ctrl+r to ask.")
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
        try:
            text = "\n".join(repl.format_answers(self.ask(state, questions)["answers"]))
        except Exception as exc:
            log.exception("ask failed")
            text = f"error: {exc}"
        self.app.call_from_thread(self.show, text)

    def action_close(self) -> None:
        self.dismiss(None)
