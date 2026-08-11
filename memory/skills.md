# Skills

tags: skills, agent, tools

- Use the tool-calling loop: read/glob/grep before editing; write minimal diffs.
- After self-editing code, run the gates (pytest/ruff/mypy) — keep them green.
- Prefer appending to memory files over rewriting them (memory_write appends by default).
- Recall now uses term-frequency + fuzzy scoring; query with concrete nouns for best hits.
