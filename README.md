# olla-jev

A local server that runs **System One decision models** from Hugging Face behind TypeSafe's
**Jev / System One** wire API — the way Ollama runs LLMs.

> **Already calling TypeSafe or Jev? This is a drop-in replacement.** Point `TYPESAFE_BASE_URL` at
> this server and the stock `typesafe-sdk` keeps working: same routes, same request and response
> shapes, no API key. The answers come from a model on your machine instead of the hosted service.

- A decision model writes no text. You send one **state** and any number of typed **questions**.
- You get back a probability for each question: `noul` (yes/no), `choice` (pick one option) or
  `score` (expected level on an ordered rubric).
- Pick a model with the request's `model` field. Models load on first use and unload when idle.
- Commands mirror Ollama's: `serve`, `run`, `pull`, `list`, `ps`, `show`, `rm`, `stop`, `cp`.

olla-jev is an independent project. It is not affiliated with or endorsed by Ollama or TypeSafe.

## Install

macOS and Linux:

```sh
curl -fsSL https://raw.githubusercontent.com/nvkudva/olla-jev/main/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/nvkudva/olla-jev/main/install.ps1 | iex
```

- The script installs [uv](https://docs.astral.sh/uv/) if it is missing, then installs `olla-jev`
  as a command in its own Python 3.12 environment (`uv tool install`).
- Add `--service` (`… | sh -s -- --service`) to also run the server in the background at every
  login: a launchd agent on macOS, a systemd user unit on Linux.
- Already have uv? `uv tool install git+https://github.com/nvkudva/olla-jev` does the same.
- Uninstall with `curl -fsSL https://raw.githubusercontent.com/nvkudva/olla-jev/main/install.sh | sh -s -- --uninstall`
  (Windows: download `install.ps1` and run `.\install.ps1 -Uninstall`). It removes the command and
  the service and keeps your config and downloaded models.

## Quick start

```sh
olla-jev            # first run: pick a model, device and address; it downloads and serves
olla-jev run        # in another terminal: ask the model questions
```

- The setup screen lists the curated models, all under 4 GB. Run `olla-jev setup` to change your
  choices later.
- The demo page opens at <http://127.0.0.1:8000/demo>.

## Example

```sh
curl -s http://127.0.0.1:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
  "model": "Mapika/decider-4b-GGUF:Q4_K_M",
  "questions": {
    "area":    {"type":"choice","instructions":"Which product area is this about?","criteria":{"refund & dispute":"A billing dispute or refund request","card":"Anything about a card","other":null}},
    "urgency": {"type":"score","instructions":"How urgent is this message?","criteria":["Can wait","Needs attention this week","Needs attention today"]},
    "refund":  {"type":"noul","instructions":"The customer is asking for a refund."}
  }
}'
```

```json
{
  "model": "Mapika/decider-4b-GGUF:Q4_K_M",
  "answers": {
    "area":    {"type":"choice","choice":"refund & dispute","confidence":0.9642,
                "probabilities":{"refund & dispute":0.9761,"card":0.0104,"other":0.0135}},
    "urgency": {"type":"score","score":1.712,"confidence":0.568,
                "legend":{"0":"Can wait","1":"Needs attention this week","2":"Needs attention today"},
                "probabilities":{"0":0.0099,"1":0.2682,"2":0.7219}},
    "refund":  {"type":"noul","noul":0.9433,"confidence":0.8866}
  },
  "usage": {"input_tokens": 198, "output_tokens": 0}
}
```

Leave `model` out, or send `jev-latest` (typesafe-sdk's default), to use the default model.

## Models

A model name is its Hugging Face repo id. Repos with several quantized files take Ollama's tag
syntax: `<repo>:<quant>` (case-insensitive) or `<repo>:<file.gguf>`. Without a tag, Q4_K_M is used.
An `hf.co/` prefix is accepted, so names copied from an Ollama command work.

The setup screen offers these, all under 4 GB:

| Model | Download | Runs on | Languages | Limits |
|---|---|---|---|---|
| `Mapika/decider-4b-GGUF:Q4_K_M` (default) | 2.7 GB | llama.cpp (Metal) | English | 255 options, 10 levels, 32k tokens |
| `Mapika/decider-2b-GGUF:Q4_K_M` | 1.2 GB | llama.cpp | English | same |
| `Mapika/decider-2b-GGUF:Q8_0` | 2.0 GB | llama.cpp | English | same |
| `Mapika/decider-2b` | 3.8 GB | PyTorch | English | same |
| `Mapika/decider-0.8b` | 1.5 GB | PyTorch | English | same |
| `convaiinnovations/laya` | 0.85 GB | PyTorch | English | 512 tokens |
| `convaiinnovations/laya-multilingual` | 0.68 GB | PyTorch | 100+ | 1024 tokens |
| `convaiinnovations/laya-typed-decisions` | 0.85 GB | PyTorch | English | 1024 tokens |
| `SupersonicLabs/Julia-1` | 0.57 GB | PyTorch (CPU) | Multilingual | **2–20 options**, 8k tokens |
| `com-kotobalabs/open-jev-deberta-v3-large` | 1.7 GB | PyTorch | English | **512 tokens** |
| `jaredpalmer/kev-0.5b`, `kev-0.6b`, `kev-0.8b` | 1–1.7 GB with base | PyTorch | English | 255 options, 8k tokens |
| `internlm/Intern-Decision-0.8B` | 1.7 GB | PyTorch | Multilingual | 62 options, 16 questions |
| `llm-semantic-router/Decision-1.0-Kai-0.6B`, `-Lex-0.6B` | 2.3 GB | PyTorch | English | 255 options, **1024 tokens** |

Any other repo works when it belongs to one of these families (decider, laya, julia, open-jev, kev,
intern-decision, decision1), for example a fine-tune or a bigger size. Requests over a model's
limits get a 422 before the model runs. `olla-jev show <model>` prints them.

### Repo code and trust

Julia, open-jev, Intern-Decision and Decision-1.0 run Python code shipped in the model repo, with
your user's privileges. The first `pull` of such a repo shows the commit and its code files and
asks you to trust that exact commit (`--trust` skips the question). Every repo is pinned to the
commit of its first download and never updates by itself.

kev's loader is vendored from GitHub at a pinned commit (`olla_jev/_vendor/kev`), and its `head.pt`
is loaded with `torch.load(weights_only=True)`, so the file cannot run code.

## Commands

| Command | What it does |
|---|---|
| `olla-jev` / `serve [model]` | start the server; the first run opens setup |
| `setup` | pick the default model, device and address, then serve |
| `run [model]` | ask questions from the terminal |
| `pull <model>… [--trust]` | download models |
| `list` | downloaded models (`*` marks the default) |
| `ps` | loaded models, device and unload time |
| `show <model>` | family, pinned commit, limits, path |
| `rm <model>…` | delete a download (one quant of a GGUF repo, or the whole repo) |
| `stop <model>` | unload a model now |
| `cp <source> <name>` | give a model a short name |
| `service install\|uninstall\|status\|logs` | run the server in the background at login |

Every command has `--help` with an example.

`run` uses the running server, or loads the model in its own process when none is running:

```
state> I was charged twice for order 8841 and want a refund.
q1> noul: The customer asks for a refund.
q2> choice: Which team? | billing, support, sales
q3>
  q1         noul    0.943
  q2         choice  billing   (billing 0.95  support 0.03  sales 0.02)
```

## API

- **Jev / System One:** `GET /v1/models`, `POST /v1/systemone`. Bearer headers are accepted and
  ignored.
- **Model management, Ollama style:** `GET /api/tags`, `GET /api/ps`, `POST /api/pull`
  (`{"model", "stream", "trust"}`, NDJSON progress), `POST /api/show`, `DELETE /api/delete`,
  `POST /api/copy`, `POST /api/stop`.
- **Demo page:** `/demo` — a request editor with a model picker, five ready-made examples and a log
  of answers. `/ui/presets` serves the examples.

Every error is `{"detail": [{"loc", "msg", "type"}]}`: 404 `model_not_found`, 403
`model_not_trusted`, 422 for an invalid request or one over the model's limits.

Confidence uses TypeSafe's formulas for every model, so it means the same thing whichever model
answered: choice `(n·p_max − 1)/(n − 1)`, score one minus the normalised expected distance from the
most likely level, noul the same as a two-option choice.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OLLAJEV_HOST` | `127.0.0.1:8000` | bind address for `serve`, and where the other commands look for it |
| `OLLAJEV_KEEP_ALIVE` | `5m` | how long an idle model stays loaded (`300`, `5m`, `1h`, `-1` = forever) |
| `OLLAJEV_MAX_LOADED_MODELS` | `1` | models in memory at once; the least recently used one unloads |
| `OLLAJEV_MODELS` | Hugging Face cache | where weights are stored |
| `OLLAJEV_DEVICE` | best available | force `cpu`, `mps` or `cuda` |
| `OLLAJEV_HOME` | OS config folder | config (default model, pins, trusted commits, aliases) and logs |

The model `serve` preloads stays loaded until the server stops.

| | macOS | Linux | Windows |
|---|---|---|---|
| Config | `~/Library/Application Support/olla-jev` | `~/.config/olla-jev` | `%LOCALAPPDATA%\olla-jev` |
| Logs | `~/Library/Logs/olla-jev` | `~/.local/state/olla-jev/log` | `%LOCALAPPDATA%\olla-jev\Logs` |
| Models | `~/.cache/huggingface/hub` | same | same |

## Known limits

- Phase 1 covers models under 4 GB. Bigger ones (decider-4b bf16, kev-4b and up, Nimble-9B …)
  are planned; see `PLAN.md`.
- Requests to one model run one at a time. On Apple GPUs concurrent forwards crash the process.
- Julia-1 runs on CPU (its runtime does not move inputs to the Apple GPU); it is fast there.
- Decision-1.0 Kai returned near-uniform `score` distributions in our tests; its `choice` and
  `noul` answers, and Lex's scores, look normal. The cause is not known yet.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). In short:

```sh
uv sync
uv run olla-jev --help
uv run pytest
```

## License

Apache-2.0. kev's loader is vendored under its Apache-2.0 license; see [NOTICE](NOTICE).
