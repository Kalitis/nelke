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

## Frontends

All four frontends share one core. Each surfaces the same chat, self-improvement
cycle, and human review gate.

```bash
nelke web [--host 127.0.0.1] [--port 8000]   # FastAPI + SSE chat/review UI
nelke tui                                     # Textual terminal UI (chat/improve/memory)
nelke bot                                     # Telegram bot (/chat /improve /cancel /memory)
```

The web frontend streams tokens over SSE; the human gate is a `/review/<id>`
page with approve/reject buttons. The TUI shows cycle events live and opens a
review modal. The Telegram bot edits messages as the answer streams and sends
inline ✅/❌ buttons for the human gate. All resolve the same `review_requests`
row — the first frontend to resolve wins.

### Chats and cycles history

Both the web and TUI frontends manage **multiple named chats**: each chat keeps
its own persisted transcript (in SQLite `sessions`/`messages`, including tool
calls) and its own per-chat memory store (`memory/chats/<session_id>/`) that the
agent reads via `recall`/writes via `memory_write`. Opening a chat reloads its
history so conversations continue across restarts. Self-improvement **cycles**
live in a separate view — the web `/cycles` page and the TUI "Improve" tab list
every cycle with its steps, timeline events and review links.

Telegram needs `NELKE_TELEGRAM_TOKEN` in `~/.nelke/.env` (see `.env.example`);
web host/port come from `NELKE_WEB_HOST`/`NELKE_WEB_PORT`. If api.telegram.org
is blocked from your network, set `NELKE_TELEGRAM_PROXY` to your local proxy
(e.g. `socks5h://127.0.0.1:12334`); the bot routes its traffic through it.

## Architecture

```
src/nelke/
├── main.py            # Typer entry
├── config.py          # pydantic-settings + provider profiles
├── core/
│   ├── agent.py       # planning + tool-calling loop + streaming
│   ├── llm.py         # multi-provider client (+ ReAct fallback parser)
│   ├── tools/         # base, registry, fs, shell, web, memory, subagent, selfedit
│   ├── memory.py      # markdown memory + INDEX + recall
│   ├── reviewer.py    # read-only AI reviewer
│   ├── cycle.py       # self-improvement cycle engine
│   ├── services.py    # shared frontend wiring (chat/cycle/review helpers)
│   ├── governance.py  # tests/lint/typecheck gate + boot check
│   ├── gitops.py      # git wrappers (subprocess)
│   └── db.py          # SQLite: sessions, cycles, steps, review_requests
├── frontends/
│   ├── cli.py          # CLI (Typer + Rich)
│   ├── web.py          # FastAPI + Jinja2 + SSE
│   ├── tui.py          # Textual TUI
│   └── telegram_bot.py # aiogram Telegram bot
├── templates/          # Jinja2 templates (web) — single source of truth
└── static/             # CSS/JS (web)
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
