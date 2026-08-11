"""Tool tests: workspace scoping, shell/python_run safety, memory tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from nelke.core.memory import MemoryStore
from nelke.core.tools import fs as fs_tools
from nelke.core.tools import memory as mem_tools
from nelke.core.tools import shell as shell_tools
from nelke.core.tools.base import ToolError


@pytest.fixture
def workspace(tmp_path) -> Path:
    return tmp_path / "ws"


async def test_read_edit_write_roundtrip(workspace):
    workspace.mkdir()
    write = fs_tools.WriteFileTool(workspace)
    r1 = await write.execute(path="notes/a.md", content="# A\n\nhello")
    assert r1.ok
    read = fs_tools.ReadFileTool(workspace)
    r2 = await read.execute(path="notes/a.md")
    assert "hello" in r2.output
    edit = fs_tools.EditFileTool(workspace)
    r3 = await edit.execute(path="notes/a.md", old_string="hello", new_string="world")
    assert r3.ok
    r4 = await read.execute(path="notes/a.md")
    assert "world" in r4.output


async def test_write_refused_outside_workspace(workspace, tmp_path):
    workspace.mkdir()
    write = fs_tools.WriteFileTool(workspace)
    with pytest.raises(ToolError):
        await write.execute(path=str(tmp_path / "evil.txt"), content="x")


async def test_read_refused_outside_workspace(workspace, tmp_path):
    workspace.mkdir()
    evil = tmp_path / "secret.txt"
    evil.write_text("secret", encoding="utf-8")
    read = fs_tools.ReadFileTool(workspace)
    with pytest.raises(ToolError):
        await read.execute(path=str(evil))


async def test_glob_and_grep(workspace):
    workspace.mkdir()
    (workspace / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (workspace / "b.txt").write_text("foo bar", encoding="utf-8")
    globbed = await fs_tools.GlobTool(workspace).execute(pattern="**/*.py")
    assert "a.py" in globbed.output
    grep = await fs_tools.GrepTool(workspace).execute(pattern="foo", include="*.txt")
    assert "b.txt:1" in grep.output


async def test_bash_tool(workspace):
    workspace.mkdir()
    r = await shell_tools.BashTool(workspace).execute(command="echo nelke")
    assert r.ok
    assert "nelke" in r.output


async def test_python_run_blocks_dangerous_imports(workspace):
    workspace.mkdir()
    r = await shell_tools.PythonRunTool(workspace).execute(script="import os; print(os.getcwd())")
    assert not r.ok
    assert "blocked" in r.error.lower()


async def test_python_run_executes_math(workspace):
    workspace.mkdir()
    r = await shell_tools.PythonRunTool(workspace).execute(script="print(6 * 7)")
    assert r.ok
    assert "42" in r.output


async def test_python_run_utf8_output(workspace):
    """Unicode outside cp1251 (emoji/cyrillic) must round-trip via UTF-8 env."""
    workspace.mkdir()
    r = await shell_tools.PythonRunTool(workspace).execute(
        script="print('привет ✅ мир')"
    )
    assert r.ok
    assert "привет ✅ мир" in r.output


async def test_bash_utf8_output(workspace):
    """bash output is decoded from raw bytes with UTF-8 first, cp1251 fallback."""
    workspace.mkdir()
    # Windows cmd's `echo` mangles non-ASCII; route through a python one-liner
    # that emits UTF-8 (forced via PYTHONUTF8) to exercise byte decoding.
    cmd = "python -c \"import sys; sys.stdout.write('привет ✅')\""
    r = await shell_tools.BashTool(workspace).execute(command=cmd)
    assert r.ok
    assert "привет ✅" in r.output


async def test_web_fetch_content_type_guard(monkeypatch):
    """web_fetch refuses to parse non-HTML content-types (anti-bot PDF/image walls)."""
    from nelke.core.tools import web as web_mod

    captured = {}

    class FakeResp:
        headers = {"content-type": "application/pdf"}
        status_code = 200

        @property
        def request(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            captured["url"] = url
            return FakeResp()

    monkeypatch.setattr(web_mod.httpx, "AsyncClient", lambda *a, **k: FakeClient())
    tool = web_mod.WebFetchTool(timeout=5)
    r = await tool.execute(url="https://example.com/doc.pdf")
    assert not r.ok
    assert "content-type" in r.error.lower()


async def test_web_fetch_wikipedia_fallback(monkeypatch):
    """web_fetch falls back to a mirror/plain-text when the primary 403s."""
    from nelke.core.tools import web as web_mod

    urls_hit = []

    class FakeResp:
        headers = {"content-type": "text/plain; charset=utf-8"}
        status_code = 200
        content = b"# Article\n\nbody text"

        @property
        def request(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            urls_hit.append(url)
            if "api/rest_v1/page/plain/" not in url:
                resp = FakeResp()
                resp.status_code = 403
                return resp
            return FakeResp()

    monkeypatch.setattr(web_mod.httpx, "AsyncClient", lambda *a, **k: FakeClient())
    tool = web_mod.WebFetchTool(timeout=5)
    r = await tool.execute(url="https://en.wikipedia.org/wiki/Nelke")
    assert r.ok, r.error
    assert "Article" in r.output
    # primary + at least one fallback was attempted
    assert len(urls_hit) >= 2
    assert urls_hit[0] == "https://en.wikipedia.org/wiki/Nelke"


async def test_web_fetch_strips_boilerplate():
    """HTML→markdown drops scripts/styles/nav so the result is clean content."""
    from nelke.core.tools import web as web_mod

    html = (
        "<html><head><script>var x = 1;</script><style>.a{}</style></head>"
        "<body><nav><a href='/x'>Home</a></nav>"
        "<h1>Title</h1><p>Body text</p>"
        "<footer>© 2026</footer></body></html>"
    )
    md = web_mod._html_to_markdown(html)
    assert "Title" in md
    assert "Body text" in md
    assert "var x" not in md
    assert ".a{" not in md
    assert "nav" not in md.lower() or "Home" not in md


async def test_rewrite_wikiwand_keeps_lang():
    """wikiwand fallback maps <lang>.wikiwand.com/<title> → www/<lang>/<title>."""
    from nelke.core.tools import web as web_mod

    out = web_mod._rewrite_wikiwand("https://ru.wikiwand.com/Nelke")
    assert out == "https://www.wikiwand.com/ru/Nelke"
    # www canonical form is a no-op (already canonical)
    assert web_mod._rewrite_wikiwand("https://www.wikiwand.com/en/Nelke") == "https://www.wikiwand.com/en/Nelke"


async def test_memory_recall_and_index(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    store.write("skills.md", "# Skills\n\nUse the tool loop for safety.", overwrite=True)
    store.write("facts/llm.md", "# Llms\n\nLocal models may lack function calling.", overwrite=True)
    index = store.build_index()
    assert "# Nelke Memory Index" in index
    assert "skills" in index.lower()
    hits = store.recall("tool loop", top_k=5)
    assert hits
    assert hits[0].name == "skills.md"
    assert "safety" in hits[0].snippet


async def test_memory_list_and_show(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    store.write("skills.md", "# Skills\n\nUse the tool loop for safety.\ntags: skills", overwrite=True)
    store.write("facts/llm.md", "# LLMs\n\nLocal models may lack function calling.", overwrite=True)
    listed = await mem_tools.MemoryListTool(store).execute()
    assert {"skills.md", "facts/llm.md"} <= set(listed.output.splitlines())
    shown = await mem_tools.MemoryShowTool(store).execute(path="skills.md")
    assert shown.ok
    assert "# Skills" in shown.output
    assert "tool loop" in shown.output


async def test_memory_show_missing_file(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    r = await mem_tools.MemoryShowTool(store).execute(path="nope.md")
    assert not r.ok
    assert "nope.md" in r.error


async def test_memory_show_refuses_escape(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    r = await mem_tools.MemoryShowTool(store).execute(path="../evil.md")
    assert not r.ok


async def test_memory_write_tool_appends_and_rebuilds(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    tool = mem_tools.MemoryWriteTool(store)
    r1 = await tool.execute(path="lessons.md", content="- lesson one")
    assert r1.ok
    r2 = await tool.execute(path="lessons.md", content="- lesson two")
    assert r2.ok
    content = store.read("lessons.md")
    assert "- lesson one" in content and "- lesson two" in content
    assert (tmp_path / "memory" / "INDEX.md").exists()


async def test_memory_write_refuses_outside_dir(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    tool = mem_tools.MemoryWriteTool(store)
    r = await tool.execute(path="../evil.md", content="x")
    assert not r.ok
