"""Memory store tests: INDEX generation, keyword recall, and semantic embeddings."""

from __future__ import annotations

from nelke.core.embeddings import Embedder, LocalEmbedder
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


def test_recall_semantic_no_token_overlap(tmp_path):
    """Embedding similarity surfaces files that share meaning but no exact words."""
    store = _store(tmp_path)
    store.write("facts/git.md", "# Git\n\nRecover a removed branch with git checkout from reflog.", overwrite=True)
    store.write("facts/web.md", "# Web\n\nHTTP caching and retries.", overwrite=True)
    # "undelete a lost branch" shares tokens with git? No—but it shares *meaning*
    # with the git file (recover/removed/branch). The semantic pass should rank
    # git.md above web.md even without word overlap.
    hits = store.recall("undelete a deleted git branch", top_k=5)
    names = [h.name for h in hits]
    assert names
    assert names[0] == "facts/git.md"


def test_auto_link_related_files(tmp_path):
    """auto_link() appends Related: cross-references between similar files."""
    store = MemoryStore(tmp_path / "memory")
    store.write("a.md", "# Alpha\n\nPlan the roadmap before coding the roadmap plan stages.", overwrite=True)
    store.write("b.md", "# Beta\n\nRoadmap planning stages before coding.", overwrite=True)
    store.write("c.md", "# Gamma\n\nCompletely unrelated astronomy facts about stars.", overwrite=True)
    updated = store.auto_link()
    # a and b are very similar -> cross-linked; c stays untouched.
    assert "a.md" in updated
    assert "b.md" in updated
    assert "c.md" not in updated
    assert "Related: [Beta](b.md)" in store.read("a.md")
    assert "Related: [Alpha](a.md)" in store.read("b.md")
    # idempotent: running again produces no further changes
    assert store.auto_link() == []


def test_auto_link_skips_unrelated(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    store.write("x.md", "# X\n\nOne topic entirely about baking bread.", overwrite=True)
    store.write("y.md", "# Y\n\nA different topic about quantum physics equations.", overwrite=True)
    assert store.auto_link() == []


def test_embed_cosine_similarity():
    from nelke.core.memory import _embed_similarity

    # Same-ish text -> high similarity; unrelated -> low.
    high = _embed_similarity("retrieve a deleted branch from git", "recover a removed branch with git")
    low = _embed_similarity("retrieve a deleted branch from git", "bake sourdough bread with flour")
    assert high > 0.4
    assert low < high


class _CountingLocal(LocalEmbedder):
    """LocalEmbedder that records how many times embed() was actually called."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_calls = 0

    async def embed(self, texts):
        self.embed_calls += 1
        return await super().embed(texts)

    def was_used(self):
        return self.embed_calls > 0


class _Unavailable(Embedder):
    name = "unavailable"

    async def ensure_ready(self):
        return False

    async def embed(self, texts):
        raise AssertionError("embed must not run when unavailable")


async def test_arecall_runs_real_embedder(tmp_path):
    """arecall() drives the injected embedder (not the sync local path)."""
    store = MemoryStore(tmp_path / "memory", embedder=_CountingLocal())
    store.write("facts/git.md", "# Git\n\nRecover a removed branch with git checkout from reflog.", overwrite=True)
    store.write("facts/web.md", "# Web\n\nHTTP caching and retries.", overwrite=True)
    hits = await store.arecall("undelete a deleted git branch", top_k=5)
    assert hits
    assert hits[0].name == "facts/git.md"
    assert store._embedder.was_used()


async def test_arecall_falls_back_when_embedder_unavailable(tmp_path):
    """An unavailable embedder degrades to the local sync recall."""
    emb = _Unavailable()
    store = MemoryStore(tmp_path / "memory", embedder=emb)
    store.write("facts/git.md", "# Git\n\nRecover a branch with git.", overwrite=True)
    hits = await store.arecall("git", top_k=5)
    assert hits and hits[0].name == "facts/git.md"
    assert emb.available is False


async def test_arecall_uses_embedder_threshold(tmp_path):
    """arecall applies embedder.min_sim for the no-lexical-overlap gate."""
    emb = _CountingLocal()
    store = MemoryStore(tmp_path / "memory", embedder=emb)
    store.write("a.md", "# A\n\nCompletely unrelated astronomy facts about distant stars.", overwrite=True)
    store.write("b.md", "# B\n\nRoadmap planning stages before coding the roadmap.", overwrite=True)
    # The semantic pass is what lets a no-token-overlap query through.
    hits = await store.arecall("некий запрос про звёзды и астрономию", top_k=5)
    assert isinstance(hits, list)


async def test_auto_link_async_runs_through_embedder(tmp_path):
    emb = _CountingLocal()
    store = MemoryStore(tmp_path / "memory", embedder=emb)
    store.write("a.md", "# Alpha\n\nPlan the roadmap before coding the roadmap plan stages.", overwrite=True)
    store.write("b.md", "# Beta\n\nRoadmap planning stages before coding.", overwrite=True)
    store.write("c.md", "# Gamma\n\nCompletely unrelated astronomy facts about stars.", overwrite=True)
    updated = await store.auto_link_async()
    assert "a.md" in updated and "b.md" in updated
    assert "c.md" not in updated
    assert "Related: [Beta](b.md)" in store.read("a.md")
    assert emb.was_used()
    # idempotent
    assert await store.auto_link_async() == []


async def test_auto_link_async_falls_back_when_unavailable(tmp_path):
    emb = _Unavailable()
    store = MemoryStore(tmp_path / "memory", embedder=emb)
    store.write("a.md", "# Alpha\n\nPlan the roadmap before coding the roadmap plan stages.", overwrite=True)
    store.write("b.md", "# Beta\n\nRoadmap planning stages before coding.", overwrite=True)
    updated = await store.auto_link_async()
    assert "a.md" in updated and "b.md" in updated
