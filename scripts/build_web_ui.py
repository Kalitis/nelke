"""Build the Vite SPA into ``static/dist``.

Run directly (``python scripts/build_web_ui.py``) from the repo root before
publishing a wheel. Equivalent to ``cd web-ui && npm ci && npm run build`` —
the Vite config already emits into ``../static/dist``. Exits non-zero if the
build fails so this can be wired into a release script or CI step.

A hatchling ``BuildHookInterface`` subclass (``NelkeWebUiBuildHook``) is also
exposed for projects that want to wire the build into their wheel assembly;
register it as a separate plugin package and reference it under
``[tool.hatch.build.hooks.<name>]``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _node_available() -> bool:
    return shutil.which("npm") is not None


def _try_build(web_ui_dir: Path) -> bool:
    """Run ``npm ci && npm run build``; return True on success."""
    if not web_ui_dir.is_dir():
        return False
    try:
        subprocess.run(
            ["npm", "ci", "--no-audit", "--no-fund"],
            cwd=str(web_ui_dir),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["npm", "run", "build"],
            cwd=str(web_ui_dir),
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class NelkeWebUiBuildHook(BuildHookInterface[Any]):
    """Build the SPA before the wheel is assembled."""

    PLUGIN_NAME = "nelke-web-ui"

    def initialize(self, _version: str, build_data: dict[str, Any]) -> None:
        if os.environ.get("NELKE_SKIP_WEB_BUILD"):
            # Explicit opt-out: trust whatever is already on disk (or nothing).
            return

        root = Path(self.root)
        web_ui = root / "web-ui"
        static_dist = root / "static" / "dist"

        # Vite emits into ../static/dist; if it's already there, nothing to do.
        if static_dist.is_dir() and (static_dist / "index.html").is_file():
            return

        if not _node_available():
            # No Node on the build host: fall back to whatever exists. The
            # legacy Jinja2 UI remains functional from the bundled templates/.
            return

        if _try_build(web_ui) and static_dist.is_dir():
            return

        # Build failed — surface a loud warning but never abort the wheel:
        # the Python package still works (with the legacy UI).
        import sys

        print(
            "[nelke-web-ui] WARNING: SPA build failed; wheel will only contain "
            "the legacy Jinja2 UI.",
            file=sys.stderr,
        )


def main() -> int:
    """CLI entry: build the SPA from the repo root; 0 on success."""
    web_ui = Path(__file__).resolve().parent.parent / "web-ui"
    static_dist = web_ui.parent / "static" / "dist"
    if static_dist.is_dir() and (static_dist / "index.html").is_file():
        print("[nelke-web-ui] static/dist already built; nothing to do.")
        return 0
    if not _node_available():
        print("[nelke-web-ui] ERROR: npm not found on PATH.", flush=True)
        return 1
    if _try_build(web_ui):
        print("[nelke-web-ui] build OK.")
        return 0
    print("[nelke-web-ui] ERROR: build failed.", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
