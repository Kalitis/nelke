"""Tools for querying self-improvement cycles from a normal-mode (chat) agent.

The regular chat agent gets :class:`CyclesTool` so a user can talk *about* the
self-improvement happening in the background without interrupting it — the tool
reads the persisted ``cycle_events``/``cycles``/``cycle_steps`` tables and
returns a human-readable progress trace. It never mutates anything.
"""

from __future__ import annotations

from typing import Any

from nelke.core.db import Database
from nelke.core.tools.base import BaseTool, ToolResult


class CyclesTool(BaseTool):
    name = "cycles"
    description = (
        "Query in-flight/finished self-improvement cycles. Use to answer questions "
        "about what a cycle is doing (progress, tool calls, files edited, gate "
        "results). Returns a recent progress trace. Read-only."
    )
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Number of most recent cycle events to show per cycle",
                "default": 40,
            },
        },
        "required": [],
    }

    def __init__(self, db: Database, default_limit: int = 40) -> None:
        self.db = db
        self.default_limit = default_limit

    async def execute(self, **kwargs: Any) -> ToolResult:
        limit = int(kwargs.get("limit") or self.default_limit)
        # Most recent cycles first; status + objective + current status.
        cycles = list(self.db.list_cycles())
        if not cycles:
            return ToolResult.success("no self-improvement cycles have run yet")

        lines: list[str] = []
        for cyl in cycles[:3]:
            cid = cyl["id"]
            lines.append(
                f"cycle {cid}\n"
                f"  objective: {cyl['objective']}\n"
                f"  branch: {cyl['branch']}\n"
                f"  status: {cyl['status']}  (AI: {cyl['ai_verdict']} / human: {cyl['human_verdict']})"
            )
            events = self.db.list_cycle_events(cid, limit=limit)
            if not events:
                lines.append("  (no progress events recorded yet)")
                continue
            lines.append("  progress:")
            for ev in events:
                kind = ev["kind"]
                msg = (ev["message"] or "").strip()
                if kind == "agent_token":
                    continue  # token deltas are too noisy for a trace
                if kind in {"agent_tool", "agent_tool_result"}:
                    payload = self._payload(ev["payload"])
                    tool = payload.get("tool", "")
                    if kind == "agent_tool":
                        args = payload.get("args") or {}
                        args_txt = ", ".join(
                            f"{k}={str(v)[:30]}" for k, v in list(args.items())[:2]
                        )
                        lines.append(f"    - [step {ev['seq']}] tool {tool}({args_txt})")
                    else:
                        lines.append(f"    - [step {ev['seq']}] -> {tool}: {payload.get('snippet', '')[:120]}")
                    continue
                # No `step` column on cycle_events; read it from the payload
                # (where the emitter puts step=step_no) and fall back to seq.
                step = self._payload(ev["payload"]).get("step") or ev["seq"]
                lines.append(f"    - [step {step}] {kind}: {msg[:160]}")
        return ToolResult.success("\n".join(lines))

    @staticmethod
    def _payload(raw: Any) -> dict[str, Any]:
        import json

        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except (ValueError, TypeError):
            return {}
