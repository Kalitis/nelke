# LLM providers

tags: facts, llm, providers

- Nelke talks to any OpenAI-compatible endpoint via the `openai` library.
- Profiles live in `~/.nelke/config.toml`; secrets come from env vars via `api_key_ref`.
- LM Studio: http://localhost:1234/v1 ; Ollama: http://localhost:11434/v1 ; keys can be dummy.

## Prompt caching
- Automatic prompt caching engages on identical, long prefixes — **not gated by temperature**. Verified on dslab 2026-08-11: a repeated ~3600-token prefix reads ~92% from cache (`cache_read_tokens`) at both T=1.0 and T=0.0.
- Nelke defaults `agent_temperature` to `1.0` (Settings, `NELKE_AGENT_TEMPERATURE`) and threads it to agent, subagents, cycle-worker and reviewer. Cache is tracked in usage as `cache_read_tokens` / `cache_read_pct` (percent of prompt served from cache) and surfaced across CLI/TUI/Telegram/API. #facts llm caching temperature cost

Related: [Nelke Development Vector (2026-08-11)](chats/20260811054228-2ada9b87/development/roadmap.md)

Related: [О пользователе](chats/20260811070422-6ed4ba1d/about-user.md)

Related: [Проект Nelke](chats/20260811070422-6ed4ba1d/projects/nelke.md)

Related: [Циклы самоулучшения](chats/20260811105752-37bb9979/cycles/notes.md)

Related: [Наблюдения](chats/20260811105752-37bb9979/notes/workspace.md)

Related: [Цикл улучшения 20260811121107-6f0a032f](chats/20260811121107-6f0a032f/cycles/notes.md)

Related: [Рабочий каталог 20260811121107-6f0a032f](chats/20260811121107-6f0a032f/notes/workspace.md)

Related: [N4 / REBIRTH — рабочая конституция строителя](chats/unknown/roadmap.md)

Related: [Lessons](lessons.md)

Related: [Skills](skills.md)

Related: [Зачем существует N4 — Нова и REBIRTH](chats/unknown/rebirth-nova.md)

Related: [ML Sandbox (WSL)](facts/ml-sandbox-wsl.md)

Related: [Оценка двух подходов к долговременной памяти (steering vs merging) — 2026-08-12](chats/20260812093926-ff525969/cycles/notes.md)

Related: [N4 / Nachtschatten — долговременная память: прототипы steering / merging / associative](chats/20260812093926-ff525969/cycles/nachtschatten-memory-prototypes.md)
