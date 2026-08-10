# Nelke

Nelke is a self-improving general-purpose agent (research, math, programming, tool use) - "like Kilo".
It shares one frontend-agnostic core across CLI, web, TUI and Telegram frontends. The headline feature:
Nelke can **edit and commit into its own repository** (source code, prompts, tools, and a markdown
memory store) via a bounded **self-improvement cycle** governed by tests/lint/typecheck, an AI reviewer
agent, and a human review gate.

## Install

```bash
cd source/repos/nelke
uv sync            # create venv + install dependencies (incl. dev)
uv run nelke doctor
```

Configure a provider (see `config.example.toml` / `.env.example`):

```bash
nelke config init           # writes ~/.nelke/config.toml and ~/.nelke/.env templates
```

## Usage

```bash
nelke chat                   # interactive chat
nelke task "summarize this file"   # one-shot task
nelke improve "add a memory lesson about pytest fixtures"   # self-improvement cycle
nelke review list | approve <id> | reject <id>
nelke memory list | show <name> | edit <name> | recall <query>
nelke config show | init
nelke db status
nelke doctor
```

## Architecture

```
src/nelke/
├── main.py            # Typer entry
├── config.py          # pydantic-settings + provider profiles
└── core/
    ├── agent.py       # planning + tool-calling loop + streaming
    ├── llm.py         # multi-provider client (+ ReAct fallback parser)
    ├── tools/         # base, registry, fs, shell, web, memory, subagent, selfedit
    ├── memory.py      # markdown memory + INDEX + recall
    ├── reviewer.py    # read-only AI reviewer
    ├── cycle.py       # self-improvement cycle engine
    ├── governance.py  # tests/lint/typecheck gate + boot check
    ├── gitops.py      # git wrappers (subprocess)
    └── db.py          # SQLite: sessions, cycles, steps, review_requests
```

## Self-improvement cycle

`nelke improve "<objective>"`:

1. Branch `improve/<cycle-id>-<slug>` off `main`.
2. Working agent (self-edit tools on) edits the repo toward the objective.
3. Gate: lint + typecheck + tests must pass before each commit. Failures feed back to the agent.
4. Boot-check after every commit; a crash rolls the commit back automatically.
5. AI reviewer (read-only agent) approves or requests changes over `git diff main...branch`.
6. Human review gate on the active frontend; approval merges `--no-ff` into `main`.

## Testing

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src/nelke
```
