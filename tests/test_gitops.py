"""GitOps wrappers against a real temporary repo."""

from __future__ import annotations

import pytest

from nelke.core.gitops import GitError, GitRepo


def _repo(tmp_path) -> GitRepo:
    root = tmp_path / "r"
    root.mkdir()
    repo = GitRepo(root)
    repo.init()
    repo.configure_local_identity("Tester", "t@example.com")
    (root / "f.txt").write_text("one\n", encoding="utf-8")
    repo.add_all()
    repo.commit("first")
    return repo


def test_branch_commit_diff(tmp_path):
    repo = _repo(tmp_path)
    assert repo.current_branch() == "main"
    repo.checkout_new_branch("improve/x", base="main")
    (repo.repo / "f.txt").write_text("one\ntwo\n", encoding="utf-8")
    repo.add_all()
    sha = repo.commit("improve f", "Nelke-Self-Improve: cycle c1 step 1")
    assert sha
    diff = repo.diff("main", "HEAD")
    assert "+two" in diff
    # trailer present in commit body
    body = repo._run("log", "-1", "--format=%B").stdout
    assert "Nelke-Self-Improve: cycle c1 step 1" in body


def test_revert_commit(tmp_path):
    repo = _repo(tmp_path)
    repo.checkout_new_branch("improve/y", base="main")
    (repo.repo / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    repo.add_all()
    sha = repo.commit("bad change")
    assert (repo.repo / "bad.py").exists()
    result = repo.revert_commit(sha)
    assert result.ok
    assert not (repo.repo / "bad.py").exists()
    assert "revert" in repo.log().lower()


def test_merge_no_ff_with_trailers(tmp_path):
    repo = _repo(tmp_path)
    repo.checkout_new_branch("improve/z", base="main")
    (repo.repo / "f.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    repo.add_all()
    repo.commit("more", "Co-authored-by: Nelke <nelke@local>")
    repo.checkout("main")
    result = repo.merge_no_ff("improve/z", "merge z", "Co-authored-by: Nelke <nelke@local>")
    assert result.ok
    assert repo.current_branch() == "main"
    assert "two\nthree" in (repo.repo / "f.txt").read_text()
    # a merge commit exists (--no-ff)
    merges = repo._run("log", "--merges", "--oneline").stdout
    assert "merge z" in merges


def test_stash_all_discards_changes(tmp_path):
    repo = _repo(tmp_path)
    (repo.repo / "f.txt").write_text("dirty\n", encoding="utf-8")
    assert repo.has_changes()
    repo.stash_all()
    assert not repo.has_changes()
    assert (repo.repo / "f.txt").read_text() == "one\n"


def test_paths_changed_detects_untracked_and_modified(tmp_path):
    repo = _repo(tmp_path)
    assert not repo.paths_changed(["f.txt"])
    (repo.repo / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert repo.paths_changed(["new.py"])  # brand-new untracked file
    (repo.repo / "f.txt").write_text("one\nchanged\n", encoding="utf-8")
    assert repo.paths_changed(["f.txt"])  # modified tracked file
    assert not repo.paths_changed(["README.md"])  # absent file: unchanged


def test_checkout_missing_base_raises(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(GitError):
        repo.checkout_new_branch("improve/z", base="no-such-branch")
