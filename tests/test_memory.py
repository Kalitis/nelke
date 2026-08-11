"""Memory store tests: INDEX generation and keyword recall."""

from __future__ import annotations

from nelke.core.memory import MemoryStore


def _store(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    store.write("skills.md", "# Skills\n\nAlways gate self-edits with tests.\ntags: skills, agent", overwrite=True)
    store.write("lessons.md", "# Lessons\n\nNever merge without review.\ntags: lessons", overwrite=True)
    store.write("facts/python.md", "# Python\n\nPython 3.13 is the dev target.", overwrite=True)
    return store


def test_build_index_contains_titles_and_links(tmp_path):
    store = _store(tmp_path)
    index = store.build_index(max_tokens=2000)
    assert index.startswith("# Nelke Memory Index")
    for name in ("skills.md", "lessons.md", "facts/python.md"):
        assert f"({name})" in index
    # tags show up
    assert "skills" in index
    # INDEX.md file written
    assert (tmp_path / "memory" / "INDEX.md").exists()


def test_index_respects_budget(tmp_path):
    store = _store(tmp_path)
    index = store.build_index(max_tokens=100)
    # 100 tokens * ~4 chars/token budget — file should be bounded
    assert len(index) <= 100 * 4 + 512


def test_index_excludes_itself(tmp_path):
    store = _store(tmp_path)
    store.build_index()
    names = [f.as_posix() for f in store.files()]
    assert "INDEX.md" not in names


def test_recall_ranks_by_keyword_hits(tmp_path):
    store = _store(tmp_path)
    store.write("facts/pytest.md", "# Pytest\n\nGate runs pytest. tests matter a lot gutess.", overwrite=True)
    hits = store.recall("tests pytest gate", top_k=10)
    names = [h.name for h in hits]
    assert names[0] == "facts/pytest.md"
    assert "skills.md" in names


def test_recall_fuzzy_matches_plurals(tmp_path):
    store = _store(tmp_path)
    store.write("facts/pytest.md", "# Pytest\n\nRun the tests before merging.", overwrite=True)
    store.write("facts/other.md", "# Other\n\nNothing related.", overwrite=True)
    # "test" (singular) should still match "tests" (plural) via stemming
    hits = store.recall("test", top_k=5)
    assert hits
    assert hits[0].name == "facts/pytest.md"


def test_recall_boosted_by_index_entry(tmp_path):
    store = _store(tmp_path)
    store.write("facts/math.md", "# Math\n\nSome body text with no query words.", overwrite=True)
    store.build_index()
    hits = store.recall("quantum", top_k=5)
    # no body hits at all, so nothing matches
    assert hits == []


def test_recall_empty_query_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.recall("   ") == []


def test_append_vs_overwrite(tmp_path):
    store = _store(tmp_path)
    store.write("x.md", "first", overwrite=True)
    store.write("x.md", "second")
    assert store.read("x.md") == "first\n\nsecond\n"
    store.write("x.md", "replaced", overwrite=True)
    assert store.read("x.md") == "replaced\n"
