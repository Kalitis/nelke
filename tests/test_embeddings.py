"""Embedding backend tests: local hashing, OpenAI-compatible endpoint, config."""

from __future__ import annotations

import json

import httpx

from nelke.core.embeddings import (
    Embedder,
    EmbeddingConfig,
    LocalEmbedder,
    OpenAICompatEmbedder,
    _embed_similarity,
    build_embedder,
    load_embedding_config,
)


class TestLocalEmbedder:
    async def test_similarity_separates_topics(self) -> None:
        high = _embed_similarity("retrieve a deleted branch from git", "recover a removed branch with git")
        low = _embed_similarity("retrieve a deleted branch from git", "bake sourdough bread with flour")
        assert high > 0.4
        assert low < high

    async def test_embed_returns_fixed_dim_vectors(self) -> None:
        emb = LocalEmbedder()
        assert await emb.ensure_ready() is True
        vecs = await emb.embed(["one two three", "one two three", ""])
        assert len(vecs) == 3
        assert len(vecs[0]) == 128
        # same input -> same vector
        assert vecs[0] == vecs[1]
        # empty text still yields a well-formed vector
        assert vecs[2] == [0.0] * 128


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestOpenAICompatEmbedder:
    async def test_embed_posts_pinned_model_and_returns_vectors(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = request.url.path
            seen["body"] = json.loads(request.content)
            inputs = json.loads(request.content)["input"]
            return httpx.Response(
                200, json={"data": [{"embedding": [float(i + 1), 0.0]} for i in range(len(inputs))]}
            )

        client = _mock_client(handler)
        emb = OpenAICompatEmbedder("http://localhost:1234/v1", model="bge-m3", http_client=client)
        try:
            vecs = await emb.embed(["a", "b"])
        finally:
            await emb.close()
        assert seen["url"] == "/v1/embeddings"
        assert seen["body"] == {"model": "bge-m3", "input": ["a", "b"]}
        assert vecs == [[1.0, 0.0], [2.0, 0.0]]

    async def test_embed_caches_by_text(self) -> None:
        requests = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal requests
            requests += 1
            inputs = json.loads(request.content)["input"]
            return httpx.Response(
                200,
                json={"data": [{"index": i, "embedding": [float(i + 1)]} for i in range(len(inputs))]},
            )

        client = _mock_client(handler)
        emb = OpenAICompatEmbedder("http://localhost:1234/v1", model="m", http_client=client)
        try:
            await emb.embed(["x", "y"])
            await emb.embed(["x", "y", "x"])  # both cached
        finally:
            await emb.close()
        assert requests == 1

    async def test_auto_detects_embedding_model_from_models(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(
                    200, json={"data": [{"id": "chat-model-7b"}, {"id": "multilingual-e5-large"}]}
                )
            assert json.loads(request.content)["model"] == "multilingual-e5-large"
            return httpx.Response(200, json={"data": [{"embedding": [1.0]}]})

        client = _mock_client(handler)
        emb = OpenAICompatEmbedder("http://localhost:1234/v1", http_client=client)
        try:
            assert await emb.ensure_ready() is True
            assert emb._model == "multilingual-e5-large"
            vecs = await emb.embed(["hello"])
            assert vecs == [[1.0]]
        finally:
            await emb.close()

    async def test_unavailable_when_no_embedding_model_loaded(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"id": "qwen3:8b"}]})

        client = _mock_client(handler)
        emb = OpenAICompatEmbedder("http://localhost:1234/v1", http_client=client)
        try:
            assert await emb.ensure_ready() is False
            assert emb.available is False
            assert await emb.embed(["x"]) is None
        finally:
            await emb.close()

    async def test_http_error_degrades_to_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "model not found"})

        client = _mock_client(handler)
        emb = OpenAICompatEmbedder("http://localhost:1234/v1", model="nope", http_client=client)
        try:
            # pinned model -> ready, but the embed call fails gracefully
            assert await emb.ensure_ready() is True
            assert await emb.embed(["x"]) is None
        finally:
            await emb.close()


class TestConfig:
    def test_load_embedding_config_absent(self, tmp_path) -> None:
        assert load_embedding_config(str(tmp_path / "config.toml")) is None

    def test_load_embedding_config_parsed(self, tmp_path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[embeddings]\n"
            'base_url = "http://localhost:1234/v1"\n'
            'api_key = "lm-studio"\n'
            'model = "bge-m3"\n'
            'timeout = 5.0\n',
            encoding="utf-8",
        )
        data = load_embedding_config(str(cfg))
        assert data == EmbeddingConfig(
            base_url="http://localhost:1234/v1", api_key="lm-studio", model="bge-m3", timeout=5.0
        )

    def test_load_embedding_config_ignores_empty_model(self, tmp_path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[embeddings]\nbase_url = "http://localhost:1234/v1"\nmodel = ""\n', encoding="utf-8")
        data = load_embedding_config(str(cfg))
        assert data is not None
        assert data.model is None

    def test_build_embedder_defaults_to_local_without_config(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("NELKE_EMBEDDINGS_ENABLED", "true")
        emb = build_embedder(config=None, path=str(tmp_path / "config.toml"))
        assert isinstance(emb, LocalEmbedder)

    def test_build_embedder_uses_dense_when_configured(self, tmp_path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[embeddings]\nbase_url = "http://localhost:1234/v1"\n', encoding="utf-8")
        emb = build_embedder(config=load_embedding_config(str(cfg)))
        assert isinstance(emb, OpenAICompatEmbedder)

    def test_build_embedder_disabled_forces_local(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("NELKE_EMBEDDINGS_ENABLED", "false")
        emb = build_embedder(config=load_embedding_config(str(tmp_path)), path=str(tmp_path))
        assert isinstance(emb, LocalEmbedder)


class _CountingLocal(LocalEmbedder):
    """LocalEmbedder that records how many times embed() was actually invoked."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        return await super().embed(texts)

    def was_used(self) -> bool:
        return self.embed_calls > 0


class _UnavailableEmbedder(Embedder):
    name = "unavailable"
    available = False

    async def ensure_ready(self) -> bool:
        return False

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        raise AssertionError("embed must not be called when unavailable")
