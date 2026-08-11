"""Shell and sandboxed Python execution tools (workspace-scoped cwd).

On Windows the OS runs subprocess pipes in the legacy console codepage
(cp1251 for Cyrillic locales), so any Unicode outside that codepage — emoji,
math symbols, some Cyrillic — either raises ``UnicodeEncodeError`` during
"print" or mangles bytes coming back. Both tools therefore read the raw
bytes, decode with UTF-8 first and fall back to the ANSI codepage, and inject
``PYTHONUTF8=1`` into the environment so ``python_run`` child processes emit
UTF-8 regardless of the console locale.
"""

from __future__ import annotations

import locale
import os
import re
import subprocess
import sys
from pathlib import Path

from nelke.core.tools.base import BaseTool, ToolResult

MAX_SHELL_OUTPUT = 30_000

# Decoding order: try UTF-8 first (modern tools emit it), then the ANSI
# codepage that Windows shells actually default to, then latin-1 as a last
# resort so we never crash on an undecodable byte.
def _decode_bytes(data: bytes) -> str:
    if not data:
        return ""
    encodings = ["utf-8"]
    try:
        encodings.append(locale.getpreferredencoding(False))
    except Exception:  # noqa: BLE001
        pass
    encodings.append("latin-1")
    seen: set[str] = set()
    for enc in encodings:
        if not enc or enc in seen:
            continue
        seen.add(enc)
        try:
            return data.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def _utf8_env() -> dict[str, str]:
    """Env forcing UTF-8 so ``python``/node etc. subprocesses print Unicode even
    on a cp1251 Windows console."""
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


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
                env=_utf8_env(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(f"command timed out after {timeout}s")
        except OSError as exc:
            return ToolResult.failure(f"failed to run command: {exc}")
        out = _decode_bytes(proc.stdout or b"") + _decode_bytes(proc.stderr or b"")
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
                env=_utf8_env(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(f"python_run timed out after {timeout}s")
        except OSError as exc:
            return ToolResult.failure(f"failed to run python: {exc}")
        out = _decode_bytes(proc.stdout or b"") + _decode_bytes(proc.stderr or b"")
        if len(out) > MAX_SHELL_OUTPUT:
            out = out[:MAX_SHELL_OUTPUT] + "\n...[truncated]"
        if proc.returncode == 0:
            return ToolResult.success(out.rstrip() or "(no output)")
        return ToolResult.failure(f"exit code {proc.returncode}\n{out}".rstrip())
