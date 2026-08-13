# Nelke

Nelke is a self-improving general-purpose agent (research, math, programming, tool use) - "like Kilo".
It shares one frontend-agnostic core across CLI, web, TUI and Telegram frontends. The headline feature:
Nelke can **edit and commit into its own repository** (source code, prompts, tools, and a markdown
memory store) via a bounded **self-improvement cycle** governed by tests/lint/typecheck, an AI reviewer
agent, and a human review gate.

Self-improvement loops are **project-scoped**: each project drives its own independent
improve → verify → commit cycle that only touches files inside that project. The main
**Nelke** project is the primary loop, configured at the repo root; other projects (e.g. your
own app) get their own independent loop. No loop ever affects another project's files,
commits, or localization.

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
nelke bot                                     # Telegram bot (plain text or /chat /new /history /chats /open /improve /review /cancel /memory)
```

The web frontend streams tokens over SSE; the human gate is a `/review/<id>`
page with approve/reject buttons. The TUI shows cycle events live and opens a
review modal. The Telegram bot edits messages as the answer streams and sends
inline ✅/❌ buttons for the human gate; plain text messages are treated as
`/chat` (no prefix needed), and `/review approve|reject <id>` resolves a
pending review by text — so a cycle parked on the gate can always be approved
from Telegram even if the inline keyboard is gone. All resolve the same
`review_requests` row — the first frontend to resolve wins.

### Web UI

The chat UI is a **Vite + React + TypeScript** SPA (Tailwind CSS + Headless UI)
with a minimal dark theme, served by FastAPI. Features:

- **Streaming** responses over SSE with token-by-token rendering, markdown,
  and syntax-highlighted code blocks (copy button on each snippet).
- **Branching / swipes**: every assistant turn tracks its alternatives. Use the
  `‹ 2/3 ›` navigator under a message to switch between sibling answers.
- **Edit message**: edit any past user message — the old subtree is soft-deleted
  and a fresh answer is generated on the new branch.
- **Regenerate**: re-run an assistant answer from its parent user message.
- **Delete message**: remove a message and its descendants (soft delete keeps
  the history reachable for audit).
- Collapsible **tool-call** blocks show each tool invocation and its result.

The built bundle lives in `static/dist/`. When it is present, `/` serves the
SPA and unknown GET paths fall back to `index.html` for client-side routing.
Set `NELKE_WEB_LEGACY=1` to force the legacy Jinja2 chat UI instead.

#### Web UI development

For frontend development with hot-module reload, run the Vite dev server
alongside the API:

```bash
cd web-ui
npm install
npm run dev                 # Vite on http://localhost:5173 (proxies /api → :8000)
NELKE_WEB_DEV=1 uv run nelke web   # in another shell; UI redirects to :5173
```

Build the SPA for production (emits into `../static/dist`):

```bash
cd web-ui && npm run build
# or: python scripts/build_web_ui.py
```

Type-check and run the frontend unit tests:

```bash
cd web-ui && npm run typecheck && npm test
```

### Chats and cycles history

Chats are shared across frontends: both the web and TUI frontends manage
**multiple named chats**, and every chat is a single frontend-agnostic
conversation. Each chat keeps its own persisted transcript (in SQLite
`sessions`/`messages`, including tool calls) and its own per-chat memory store
under `memory/chats/<session_id>/`. Opening a chat reloads its history so
conversations continue across restarts. Because the same SQLite store backs
web, TUI and Telegram, the chat lists in all three show **every** conversation
(labelled with the frontend that created it: `web`/`tui`/`tg`), and any chat
is resumable from any frontend — pick up a chat you started on the web from
the Telegram bot with `/open <id>` (or vice versa), and the conversation
continues exactly where you left off. The Telegram bot keeps one persistent
session per Telegram chat by default (typing a message — or `/chat <text>` —
continues the conversation; `/new` starts a fresh chat, `/history` shows the
transcript, `/chats`/`/open <id>` list and resume older chats). In group chats
the bot only answers messages that mention it or reply to one of its own messages.
Self-improvement **cycles** live in a separate view — the web `/cycles` page
and the TUI "Improve" tab list every cycle with its steps, timeline events and
review links.

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
│   ├── services.py    # shared frontend wiring (chat/cycle/review/branch helpers)
│   ├── governance.py  # tests/lint/typecheck gate + boot check
│   ├── gitops.py      # git wrappers (subprocess)
│   └── db.py          # SQLite: sessions, cycles, steps, review_requests, message tree
├── frontends/
│   ├── cli.py          # CLI (Typer + Rich)
│   ├── web.py          # FastAPI + SSE (serves the SPA build + REST/SSE API)
│   ├── tui.py          # Textual TUI
│   └── telegram_bot.py # aiogram Telegram bot
├── templates/          # Jinja2 templates (legacy web UI + cycles/memory/review pages)
└── static/             # legacy CSS/JS + built SPA (static/dist/)
web-ui/                 # Vite + React + TS SPA source (builds into static/dist)
scripts/                # build_web_ui.py (SPA build), spa_smoke.mjs (playwright check)
```

## Self-improvement cycle

`nelke improve "<objective>"` runs a **project-scoped** loop. Each project has its
own cycle; the main **Nelke** project is the primary loop (configured at the repo
root in `examples/nelke_cycle.yaml`), and any other project gets its own independent
loop (see `examples/project_cycle.yaml`). A cycle only ever touches files inside its
own project — it never affects another project's files, commits, or localization.

The Nelke primary loop:

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
