# Lessons

tags: lessons, governance

- A self-improvement commit that cannot import/boot must be reverted — the boot check is the rollback gate.
- Never merge an unapproved change to main: AI review + human approval are required.
- Local models may lack native function calling; rely on the ReAct fallback parser.

- Windows bash/`python_run` subprocess pipes run in the legacy console codepage
  (cp1251 for Cyrillic) — Unicode outside it raises UnicodeEncodeError or mangles
  bytes. Fix: read raw bytes and decode UTF-8-first with ANSI/latin-1 fallback
  (`_decode_bytes` in tools/shell.py), and force `PYTHONUTF8=1`/`PYTHONIOENCODING`
  in the subprocess env so Python children emit UTF-8. #lessons windows encoding bash
- `web_fetch` anti-bot: mirror/fallback list keyed by hostname, content-type guard
  before parsing, per-char decode by declared charset, browser UA, timeout with
  connect timeout, and retry-once on transient transport errors. #lessons web reliability
- plan_first is now configurable via `NELKE_PLAN_FIRST` and plumbed through
  make_agent/settings/services/cli/web; it runs one extra non-tool plan call before
  the tool loop. Default stays off to avoid per-turn overhead. #lessons planning

Related: [Nelke Development Vector (2026-08-11)](chats/20260811054228-2ada9b87/development/roadmap.md)

Related: [О пользователе](chats/20260811070422-6ed4ba1d/about-user.md)

Related: [Проект Nelke](chats/20260811070422-6ed4ba1d/projects/nelke.md)

Related: [Циклы самоулучшения](chats/20260811105752-37bb9979/cycles/notes.md)

Related: [Наблюдения](chats/20260811105752-37bb9979/notes/workspace.md)

Related: [Цикл улучшения 20260811121107-6f0a032f](chats/20260811121107-6f0a032f/cycles/notes.md)

Related: [Рабочий каталог 20260811121107-6f0a032f](chats/20260811121107-6f0a032f/notes/workspace.md)

Related: [N4 / REBIRTH — рабочая конституция строителя](chats/unknown/roadmap.md)

Related: [LLM providers](facts/llms.md)

Related: [Skills](skills.md)

Related: [Зачем существует N4 — Нова и REBIRTH](chats/unknown/rebirth-nova.md)

Related: [ML Sandbox (WSL)](facts/ml-sandbox-wsl.md)

Related: [Оценка двух подходов к долговременной памяти (steering vs merging) — 2026-08-12](chats/20260812093926-ff525969/cycles/notes.md)
