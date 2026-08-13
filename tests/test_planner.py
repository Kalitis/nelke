"""Planner tests: task slicing and file-scope parsing."""

from __future__ import annotations

from nelke.core.planner import TaskSpec, _parse_tasks, fallback


def test_parse_tasks_reads_file_scope():
    """The planner JSON may carry a per-task 'files' list; the parser must lift
    it onto TaskSpec.files (normalised to forward-slash repo-relative paths)."""
    raw = (
        '{"tasks": ['
        ' {"title": "fix loop", "detail": "d1", "files": ["src/cycle.py", "tests/test_cycle.py"]},'
        ' {"title": "docs", "detail": "d2", "files": ["docs\\\\guide.md"]}'
        ']}'
    )
    specs = _parse_tasks(raw, "obj", max_tasks=6)
    assert [s.title for s in specs] == ["fix loop", "docs"]
    assert specs[0].files == ["src/cycle.py", "tests/test_cycle.py"]
    # backslashes normalised
    assert specs[1].files == ["docs/guide.md"]
    assert specs[0].has_file_scope


def test_parse_tasks_files_optional_and_deduped():
    """Missing 'files' -> empty list (no restriction). Duplicates are dropped."""
    raw = (
        '{"tasks": ['
        ' {"title": "a", "detail": "d", "files": ["x.py", "x.py", "y.py"]},'
        ' {"title": "b", "detail": "d"}'
        ']}'
    )
    specs = _parse_tasks(raw, "obj", max_tasks=6)
    assert specs[0].files == ["x.py", "y.py"]
    assert specs[1].files == []
    assert not specs[1].has_file_scope


def test_parse_tasks_non_string_files_dropped():
    """Non-string entries in a files list are silently dropped, not crash."""
    raw = (
        '{"tasks": ['
        ' {"title": "a", "detail": "d", "files": ["ok.py", 42, null, "also.py"]}'
        ']}'
    )
    specs = _parse_tasks(raw, "obj", max_tasks=6)
    assert specs[0].files == ["ok.py", "also.py"]


def test_fallback_task_has_no_file_scope():
    """The legacy whole-objective fallback has no file scope (unrestricted)."""
    specs = fallback("the whole objective")
    assert len(specs) == 1
    assert specs[0].title == "all"
    assert specs[0].files == []
    assert not specs[0].has_file_scope


def test_taskspec_prompt_lists_files_when_scoped():
    """A scoped slice advertises its allowed files in the worker prompt so the
    model knows its lane upfront."""
    scoped = TaskSpec(title="t", detail="d", files=["src/a.py", "tests/test_a.py"])
    unscoped = TaskSpec(title="t", detail="d")
    assert "ONLY edit these files" in scoped.as_prompt()
    assert "src/a.py" in scoped.as_prompt()
    assert "ONLY edit these files" not in unscoped.as_prompt()
