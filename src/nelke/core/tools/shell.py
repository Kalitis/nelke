"""Shell and sandboxed Python execution tools (workspace-scoped cwd)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from nelke.core.tools.base import BaseTool, ToolResult

MAX_SHELL_OUTPUT = 30_000
_FORBIDDEN_IMPORTS = re.compile(
    r"^\s*(?:from\s+|import\s+)(os|subprocess|shutil|socket|ctypes|multiprocessing|"
    r"pickle|marshal|importlib|pty|fcntl|winreg|msvcrt)\b",
    re.MULTILINE | re.IGNORECASE,
)


class BashTool(BaseTool):
    name = "bash"
    description = "Run a shell command in the workspace directory, capturing stdout/stderr. Has a timeout."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run"},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed", "default": None},
        },
        "required": ["command"],
    }

    def __init__(self, workspace: Path, default_timeout: int = 120) -> None:
        self.workspace = workspace
        self.default_timeout = default_timeout

    async def execute(self, **kwargs) -> ToolResult:
        command = str(kwargs.get("command", ""))
        timeout = int(kwargs.get("timeout") or self.default_timeout)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(f"command timed out after {timeout}s")
        except OSError as exc:
            return ToolResult.failure(f"failed to run command: {exc}")
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > MAX_SHELL_OUTPUT:
            out = out[:MAX_SHELL_OUTPUT] + "\n...[truncated]"
        if proc.returncode == 0:
            return ToolResult.success(out.rstrip() or "(no output)")
        return ToolResult.failure(f"exit code {proc.returncode}\n{out}".rstrip())


class PythonRunTool(BaseTool):
    name = "python_run"
    description = (
        "Run Python code in a sandboxed subprocess (sympy/numpy available). "
        "Modules like os, subprocess, shutil and socket are blocked. Returns stdout."
    )
    parameters = {
        "type": "object",
        "properties": {
            "script": {"type": "string", "description": "Python source code to execute"},
            "timeout": {"type": "integer", "description": "Seconds before the run is killed", "default": None},
        },
        "required": ["script"],
    }

    def __init__(self, workspace: Path, default_timeout: int = 120) -> None:
        self.workspace = workspace
        self.default_timeout = default_timeout

    async def execute(self, **kwargs) -> ToolResult:
        script = str(kwargs.get("script", ""))
        timeout = int(kwargs.get("timeout") or self.default_timeout)
        blocked = _FORBIDDEN_IMPORTS.findall(script)
        if blocked:
            return ToolResult.failure(
                f"blocked imports not allowed in python_run: {', '.join(sorted(set(blocked)))}"
            )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(f"python_run timed out after {timeout}s")
        except OSError as exc:
            return ToolResult.failure(f"failed to run python: {exc}")
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > MAX_SHELL_OUTPUT:
            out = out[:MAX_SHELL_OUTPUT] + "\n...[truncated]"
        if proc.returncode == 0:
            return ToolResult.success(out.rstrip() or "(no output)")
        return ToolResult.failure(f"exit code {proc.returncode}\n{out}".rstrip())
