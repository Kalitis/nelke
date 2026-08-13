"""Task planner for the parallel self-improvement cycle.

A single LLM call splits the cycle objective into up to ``max_tasks`` indepen-
dent slices. Each slice targets disjoint files/aspects so the parallel worker
agents can edit the shared working tree without colliding. The planner returns
strict JSON; any parse/validation error degrades gracefully to a single task
equal to the whole objective (the legacy single-worker behaviour).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

PLANNER_SYSTEM_PROMPT = """You are a task planner for Nelke's self-improvement \
cycle. Split the user objective into AT MOST {max_tasks} independent, \
non-overlapping subtasks that can be executed in parallel against the same \
repository.

Rules:
- Each subtask must touch DIFFERENT files or clearly separated concerns; two \
subtasks must never edit the same file.
- Prefer fewer, well-scoped subtasks over many tiny ones. If the objective is \
monolithic, return a single subtask equal to the whole objective.
- Keep each subtask small enough to complete in a handful of file edits.
- For every subtask, list the EXACT repo-relative file paths it is allowed to \
edit (source AND tests) under "files". Two subtasks must NEVER list the same \
file. Use forward slashes. If a subtask may touch any file (fallback / whole \
objective), return an empty files list.

Respond with ONLY a JSON object of this exact shape (no prose, no code fence):
{{"tasks":[{{"title":"short imperative title","detail":"1-3 sentences: what \
to change and where","files":["src/path/foo.py","tests/test_foo.py"]}}]}}

The JSON must parse with json.loads. Do not wrap it in markdown."""


@dataclass
class TaskSpec:
    """One slice of the planner's task breakdown."""

    title: str
    detail: str
    # Repo-relative file paths this worker is allowed to EDIT (source + tests).
    # Enforced by the self-edit tools via SelfEditContext.allowed_files. An empty
    # list means "no restriction" (the fallback whole-objective slice).
    files: list[str] = field(default_factory=list)

    @property
    def has_file_scope(self) -> bool:
        """True when this slice carries an explicit, non-empty file scope."""
        return bool(self.files)

    def as_prompt(self) -> str:
        """Render the task as the user-message handed to a worker agent."""
        parts = [self.title, "", self.detail]
        if self.files:
            parts.append("")
            parts.append("You may ONLY edit these files (others are out of scope):")
            parts.extend(f"- {f}" for f in self.files)
        return "\n".join(parts).strip()


def _extract_json_object(text: str) -> str:
    """Pull the first balanced ``{...}`` block out of a model response.

    Models occasionally wrap JSON in markdown fences or trailing prose despite
    the prompt; this recovers the JSON payload without parsing the whole text.
    Returns the original string if no braces are found (the caller will fail
    on json.loads and hit the fallback path).
    """
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _parse_tasks(raw: str, objective: str, max_tasks: int) -> list[TaskSpec]:
    """Parse the planner JSON; return ``[]`` on any validation failure."""
    payload = json.loads(_extract_json_object(raw))
    raw_tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(raw_tasks, list):
        return []
    specs: list[TaskSpec] = []
    seen_titles: set[str] = set()
    for item in raw_tasks:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        detail = str(item.get("detail", "")).strip()
        if not title:
            continue
        # Drop duplicates (planners sometimes repeat a slice).
        key = title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        # Optional explicit file scope. Normalised to repo-relative forward-slash
        # paths; deduped. Non-string entries are dropped. An absent/empty list
        # means "no restriction" (kept as []).
        raw_files = item.get("files", [])
        files: list[str] = []
        if isinstance(raw_files, list):
            seen_files: set[str] = set()
            for f in raw_files:
                if not isinstance(f, str):
                    continue
                norm = f.strip().replace("\\", "/")
                if norm and norm not in seen_files:
                    seen_files.add(norm)
                    files.append(norm)
        specs.append(TaskSpec(title=title, detail=detail or objective, files=files))
        if len(specs) >= max_tasks:
            break
    return specs


def fallback(objective: str) -> list[TaskSpec]:
    """Single-task plan equal to the whole objective (legacy single-worker)."""
    return [TaskSpec(title="all", detail=objective)]


async def plan_tasks(
    llm: Any,
    objective: str,
    *,
    max_tasks: int = 6,
    temperature: float | None = None,
) -> list[TaskSpec]:
    """Split ``objective`` into ``<= max_tasks`` independent subtasks.

    Uses a single non-tool LLM call. On any failure (LLM error, malformed JSON,
    empty list) degrades to ``[TaskSpec(title="all", detail=objective)]`` so the
    cycle always runs at least one worker — exactly the legacy behaviour.
    """
    if max_tasks < 1:
        max_tasks = 1
    system = PLANNER_SYSTEM_PROMPT.format(max_tasks=max_tasks)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Objective: {objective}"},
    ]
    try:
        resp = await llm.chat(messages, stream=False, temperature=temperature)
    except Exception:  # noqa: BLE001 - any planner failure falls back safely
        return fallback(objective)
    raw = (resp.content if resp else "") or ""
    raw = raw.strip()
    # Strip a markdown code fence if the model added one despite the prompt.
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw).strip()
    try:
        specs = _parse_tasks(raw, objective, max_tasks)
    except (json.JSONDecodeError, ValueError):
        return fallback(objective)
    if not specs:
        return fallback(objective)
    return specs
