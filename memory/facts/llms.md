# LLM providers

tags: facts, llm, providers

- Nelke talks to any OpenAI-compatible endpoint via the `openai` library.
- Profiles live in `~/.nelke/config.toml`; secrets come from env vars via `api_key_ref`.
- LM Studio: http://localhost:1234/v1 ; Ollama: http://localhost:11434/v1 ; keys can be dummy.

## Prompt caching (why Nelke used ~10x more tokens)
- OpenAI-compatible providers (OpenAI, OpenRouter, many proxies) enable **automatic prompt caching** only when `temperature=0`. Any non-zero temperature disables prefix caching, so every agent-loop call re-bills the whole growing prompt (~10x cost on long tool loops).
- Nelke now defaults `agent_temperature` to `0.0` (Settings, `NELKE_AGENT_TEMPERATURE`) and threads it to the agent, subagents, cycle-worker and reviewer so caching actually engages. #facts llm caching temperature cost
