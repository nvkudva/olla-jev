"""First-run setup screen: pick a model, device and address, then pull and serve."""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, OptionList, Select
from textual.widgets.option_list import Option

from . import config, store
from .catalog import CATALOG

PASTE = "__paste__"


def _downloaded() -> set[str]:
    return set(store.downloaded())


class Setup(App[dict[str, Any] | None]):
    TITLE = "olla-jev setup"
    SUB_TITLE = "System One decision models from Hugging Face"
    BINDINGS = [("escape", "quit_setup", "Quit")]
    CSS = """
    #models { height: 1fr; border: round $primary; }
    #paste { display: none; }
    #paste.shown { display: block; }
    .row { height: auto; margin-top: 1; }
    .row Label { width: 10; padding-top: 1; }
    #host { width: 24; }
    #port { width: 12; }
    #device { width: 20; }
    #hint { color: $text-muted; margin-top: 1; }
    """

    def compose(self) -> ComposeResult:
        saved = config.load()
        have = _downloaded()
        yield Header()
        with Vertical():
            yield Label("Which model should the server run? (all under 4 GB; ✓ = downloaded)")
            options = []
            for e in CATALOG:
                mark = "✓" if e.name.split(":")[0] in have else " "
                default = "  (default)" if e.name == config.DEFAULT_MODEL else ""
                options.append(
                    Option(
                        f"{mark} {e.name:<48} {e.size_gb:>5.2f} GB  {e.languages:<14} {e.description}{default}",
                        id=e.name,
                    )
                )
            options.append(Option("  Paste any Hugging Face repo id…", id=PASTE))
            yield OptionList(*options, id="models")
            yield Input(placeholder="user/repo or user/repo:Q4_K_M", id="paste")
            with Horizontal(classes="row"):
                yield Label("Device")
                yield Select(
                    [(d, d) for d in ("auto", "mps", "cuda", "cpu")],
                    value=saved.get("device", "auto"),
                    allow_blank=False,
                    id="device",
                )
                yield Label("Host")
                yield Input(saved.get("host", "127.0.0.1"), id="host")
                yield Label("Port")
                yield Input(str(saved.get("port", config.DEFAULT_PORT)), id="port", type="integer")
            with Horizontal(classes="row"):
                yield Button("Download and start", variant="primary", id="start")
                yield Button("Quit", id="quit")
            yield Label("Models that run Python from their repo ask for your trust before download.", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        models = self.query_one("#models", OptionList)
        current = config.load().get("default_model", config.DEFAULT_MODEL)
        ids = [e.name for e in CATALOG]
        models.highlighted = ids.index(current) if current in ids else 0
        models.focus()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self.query_one("#paste").set_class(event.option.id == PASTE, "shown")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id == PASTE:
            self.query_one("#paste", Input).focus()
        else:
            self.action_start()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()
        else:
            self.action_quit_setup()

    def action_quit_setup(self) -> None:
        self.exit(None)

    def action_start(self) -> None:
        models = self.query_one("#models", OptionList)
        option = models.get_option_at_index(models.highlighted or 0)
        model = self.query_one("#paste", Input).value.strip() if option.id == PASTE else option.id
        if not model:
            self.notify("Paste a repo id first", severity="error")
            return
        port = self.query_one("#port", Input).value.strip()
        self.exit(
            {
                "default_model": model,
                "device": self.query_one("#device", Select).value,
                "host": self.query_one("#host", Input).value.strip() or "127.0.0.1",
                "port": int(port) if port.isdigit() else config.DEFAULT_PORT,
            }
        )


def setup() -> bool:
    """Run the setup screen, then pull the chosen model. False when the user quits."""
    from .cli import pull

    choice = Setup().run()
    if not choice:
        return False
    try:
        pull(choice["default_model"])
    except (SystemExit, LookupError, ValueError) as exc:
        print(f"error: {exc}")
        return False
    config.update(**choice)
    print(f"==> Saved. Default model: {choice['default_model']}. Change it any time with: olla-jev setup")
    return True
