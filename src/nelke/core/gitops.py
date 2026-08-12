"""Git wrappers via the system ``git`` executable — the only sanctioned route to git.

All cycle mutations (branch/commit/revert/merge) go through here, never raw shell,
so they stay auditable and testable against a temporary repo.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitError(RuntimeError):
    pass


@dataclass
class GitResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int

    @property
    def text(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()


class GitRepo:
    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def _run(self, *args: str, check: bool = False) -> GitResult:
        try:
            proc = subprocess.run(
                ["git", "-C", str(self.repo), *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            result = GitResult(ok=False, stdout="", stderr=str(exc), returncode=1)
            if check:
                raise GitError(exc) from exc
            return result
        result = GitResult(
            ok=proc.returncode == 0,
            stdout=(proc.stdout or "").strip(),
            stderr=(proc.stderr or "").strip(),
            returncode=proc.returncode,
        )
        if check and not result.ok:
            raise GitError(f"git {' '.join(args)} failed: {result.text}")
        return result

    # Introspection
    def is_repo(self) -> bool:
        return self._run("rev-parse", "--is-inside-work-tree").ok

    def current_branch(self) -> str:
        result = self._run("branch", "--show-current")
        return result.stdout or "HEAD"

    def has_changes(self) -> bool:
        result = self._run("status", "--porcelain")
        return bool(result.stdout.strip())

    def paths_changed(self, paths: list[str]) -> bool:
        """True when any of ``paths`` is modified, staged, or untracked.

        Unlike ``git diff HEAD`` this also catches brand-new untracked files
        (e.g. a freshly added ``pyproject.toml``).
        """
        result = self._run("status", "--porcelain", "--", *paths)
        return bool(result.stdout.strip())

    def changed_paths(self) -> list[str]:
        """Working-tree paths that are modified, staged, or untracked.

        Renames/copies resolve to the new path. Returns ``[]`` on any git
        error (e.g. not a repository) so callers can degrade gracefully.
        Used by the governance ``test-gap`` check to see what the agent just
        produced in the working tree before it is committed.
        """
        result = self._run("status", "--porcelain", "-uall")
        if not result.ok or not result.stdout:
            return []
        paths: list[str] = []
        for line in result.stdout.splitlines():
            if len(line) < 4:
                continue
            path = line[3:].strip()
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            paths.append(path)
        return paths

    def stash_all(self) -> GitResult:
        """Stash tracked + untracked changes (used to discard a failed step)."""
        return self._run("stash", "-u", "--include-untracked")

    def head_sha(self) -> str:
        return self._run("rev-parse", "--short", "HEAD").stdout

    def log(self, n: int = 20) -> str:
        return self._run("log", f"-{n}", "--oneline", "--decorate").text

    # Setup
    def init(self, default_branch: str = "main") -> None:
        self._run("init", "-b", default_branch, check=True)

    def configure_local_identity(self, name: str, email: str) -> None:
        self._run("config", "user.name", name, check=True)
        self._run("config", "user.email", email, check=True)

    # Branches & checkout
    def checkout_new_branch(self, name: str, base: str = "main") -> GitResult:
        return self._run("checkout", "-b", name, base, check=True)

    def checkout(self, branch: str) -> GitResult:
        return self._run("checkout", branch, check=True)

    def branch_exists(self, name: str) -> bool:
        result = self._run("rev-parse", "--verify", "--quiet", f"refs/heads/{name}")
        return result.ok

    def delete_branch(self, name: str, *, force: bool = False) -> GitResult:
        """Delete a local branch (``-D`` when force, else ``-d``).

        Used to clean up a cycle branch after the cycle crashed or was killed,
        so the repo is not left on a stale ``improve/...`` branch.
        """
        flag = "-D" if force else "-d"
        return self._run("branch", flag, name)

    def ahead_counts(self, base: str, branch: str) -> int:
        """Number of commits on ``branch`` not reachable from ``base``.

        Used to detect stuck cycles: a ``running`` cycle whose branch carries no
        commits ahead of ``main`` can never have produced a mergeable result, so
        it is a stuck/failed run rather than an in-progress one.
        """
        result = self._run("rev-list", "--count", f"{base}..{branch}")
        if not result.ok or not result.stdout.strip().isdigit():
            return 0
        return int(result.stdout.strip())

    # Staging & commit
    def add_all(self, paths: list[str] | None = None) -> GitResult:
        return self._run("add", "--", *(paths or ["."]), check=True)

    def commit(self, message: str, *trailers: str) -> str:
        """Commit staged changes with an optional subject and trailer lines.

        Returns the short commit sha. Trailer lines are appended after a blank
        line (git-interpret-trailers format), preserving the repo's own identity.
        """
        body = message
        if trailers:
            body = message + "\n\n" + "\n".join(trailers)
        result = self._run("commit", "-m", body, check=False)
        if not result.ok or result.returncode != 0:
            raise GitError(f"commit failed: {result.text}")
        return self._run("rev-parse", "--short", "HEAD").stdout

    # Diff
    def diff(self, base: str, head: str = "HEAD", paths: list[str] | None = None) -> str:
        args = ["diff", base, head]
        if paths:
            args += ["--", *paths]
        return self._run(*args).text

    def diff_stat(self, base: str, head: str = "HEAD") -> str:
        return self._run("diff", "--stat", base, head).text

    # Rollback
    def revert_commit(self, sha: str) -> GitResult:
        """Create a revert commit undoing the given commit (used on boot-check failure)."""
        return self._run("revert", "--no-edit", sha, check=True)

    # Merge
    def merge_no_ff(self, branch: str, message: str, *trailers: str) -> GitResult:
        args = ["merge", "--no-ff", "-m", message]
        for trailer in trailers:
            args += ["-m", trailer]
        args.append(branch)
        return self._run(*args, check=True)
