"""`run`: ask a model questions from the terminal, like `ollama run`.

Uses the running server when there is one, otherwise loads the model in this process.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

HELP = """Type a state, then questions, one per line:
  noul: The customer asks for a refund.
  choice: Which team should handle this? | billing, support, sales
  score: How urgent is it? | can wait, this week, today
Empty line sends. Commands: /state to start a new state, /help, /bye."""


def parse_question(line: str) -> tuple[str, dict[str, Any]] | None:
    kind, _, rest = line.partition(":")
    kind = kind.strip().lower()
    if kind not in ("noul", "choice", "score") or not rest.strip():
        return None
    text, _, options = rest.partition("|")
    q: dict[str, Any] = {"type": kind, "instructions": text.strip()}
    opts = [o.strip() for o in options.split(",") if o.strip()]
    if kind == "choice":
        if len(opts) < 2:
            return None
        q["criteria"] = {o: None for o in opts}
    elif kind == "score":
        if len(opts) < 2:
            return None
        q["criteria"] = opts
    return kind, q


def format_answers(answers: dict[str, Any]) -> list[str]:
    lines = []
    for qid, a in answers.items():
        if a["type"] == "noul":
            lines.append(f"  {qid:<10} noul    {a['noul']:.3f}")
        elif a["type"] == "choice":
            dist = "  ".join(f"{k} {v:.2f}" for k, v in a["probabilities"].items())
            lines.append(f"  {qid:<10} choice  {a['choice']}   ({dist})")
        else:
            level = a["legend"][str(round(a["score"]))]
            lines.append(f"  {qid:<10} score   {a['score']:.2f} ≈ {level}   (confidence {a['confidence']:.2f})")
    return lines


def show(answers: dict[str, Any]) -> None:
    print("\n".join(format_answers(answers)))


def connect(
    model: str | None, say: Callable[[str], Any] = print
) -> tuple[Callable[..., dict[str, Any]], Callable[[], None]]:
    """A function that answers questions, and a function that releases it. Uses the running server when
    there is one, otherwise loads the model in this process."""
    from . import cli, normalize

    if cli.server_running():

        def ask(state: str, questions: dict[str, Any]) -> dict[str, Any]:
            body = {"state": state, "questions": questions}
            if model:
                body["model"] = model
            return cli.call("POST", "/v1/systemone", body)

        say(f"using the server at {cli.server_url()}")
        return ask, lambda: None

    from .manager import Manager

    manager = Manager()
    say("loading the model in this process (no server running)")
    slot = manager.get(model, keep_alive=-1)
    say(f"{slot.name} ready")

    def ask_local(state: str, questions: dict[str, Any]) -> dict[str, Any]:
        _, result = manager.run(model, state, questions, keep_alive=-1)
        return {"answers": normalize.answers(questions, result["answers"])}

    return ask_local, manager.unload_all


def run(model: str | None) -> None:
    ask, _ = connect(model, lambda text: print(f"==> {text}"))
    print(HELP)
    while True:
        state = input("\nstate> ").strip()
        if state in ("/bye", "/exit"):
            return
        if not state or state == "/help":
            print(HELP)
            continue
        while True:
            questions: dict[str, Any] = {}
            while True:
                line = input(f"q{len(questions) + 1}> ").strip()
                if not line:
                    break
                if line in ("/bye", "/exit"):
                    return
                if line == "/state":
                    questions = {}
                    break
                parsed = parse_question(line)
                if parsed is None:
                    print(
                        "  not a question; " + HELP.splitlines()[1].strip() + " (choice/score need 2+ options after |)"
                    )
                    continue
                questions[f"q{len(questions) + 1}"] = parsed[1]
            if not questions:
                break
            try:
                show(ask(state, questions)["answers"])
            except (SystemExit, ValueError, LookupError) as exc:
                print(f"  error: {exc}")
            again = input("more questions on this state? [Y/n] ").strip().lower()
            if again == "n":
                break
