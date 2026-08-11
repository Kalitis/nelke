"""Pluggable text embeddings for semantic memory (recall + auto-link).

Nelke ships with a dependency-free, deterministic local embedder (word/stem
hashing into a fixed-size bag, cosine between bags) that works offline and in
tests. When an OpenAI-compatible embeddings endpoint is configured — ``[embeddings]``
in ``~/.nelke/config.toml``, e.g. LM Studio serving a good *multilingual*
embedding model — recall and auto-link switch to real dense embeddings and fall
back to the local embedder the moment the endpoint is unreachable or no
embedding model is loaded.

All network access here is best-effort: memory must never fail because an
embeddings server is down. Every ``embed``/``ensure_ready`` call swallows
errors and returns ``None``/``False`` so callers can degrade cleanly.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import time
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Local embedder: word unigrams + bigrams (with a tiny stemmer) hashed into a
# fixed-size bag. Pure stdlib — no model, no network, deterministic.
_EMBED_DIM = 128
_EMBED_BITS = 12  # sha256 hex prefix -> int bucket into the vector

# Cosine threshold for the *local* embedder's notion of "related". Dense models
# carry their own thresholds (Embedder.min_sim / link_min_sim).
LOCAL_MIN_SIM = 0.35
LOCAL_LINK_MIN_SIM = 0.35

_WORD_SPLIT_RE = re.compile(r"[^\w]+")
# Stopwords dropped from queries to avoid noise matches.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "been", "was", "were", "it", "its", "that", "this", "these",
    "those", "from", "as", "by", "at", "be", "have", "has", "do", "does",
}

# Model-name signals that mark a ``/v1/models`` entry as embedding-capable, used
# by automatic model detection when ``model`` is not pinned in config. Generic
# multilingual families (BGE, E5, GTE, Jina, ...) come first so they win over
# accidental substring collisions.
_EMBED_MODEL_HINTS = (
    "bge",
    "e5",
    "gte",
    "jina",
    "minilm",
    "mxbai",
    "nomic",
    "instructor",
    "sentence-transformers",
    "sbert",
    "multilingual",
    ".m3",
    "embed",
)

# How long model-detection results are trusted before re-probing.
_DETECT_TTL = 30.0


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD_SPLIT_RE.split(text.lower()) if w and w not in _STOPWORDS]


def _simple_stem(word: str) -> str:
    """A tiny stemmer (plurals/'ing'/'ed') so inflections embed near each other."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def _embed_vector(text: str) -> list[int]:
    """Return a fixed-dim integer vector from word/stem features.

    Word unigrams + bigrams (with a simple stemmer) hashed into buckets — for
    short memory texts these separate genuinely different topics far better than
    raw character n-grams, which mostly measure character noise.
    """
    vector = [0] * _EMBED_DIM
    words = _tokenize(text)
    if not words:
        return vector
    stems = [_simple_stem(w) for w in words]
    features: list[str] = list(stems)
    features += [f"{stems[i]}|{stems[i+1]}" for i in range(len(stems) - 1)]
    for feat in features:
        bucket = (
            int(hashlib.sha256(feat.encode("utf-8")).hexdigest()[:_EMBED_BITS], 16) % _EMBED_DIM
        )
        vector[bucket] += 1
    return vector


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _embed_similarity(text_a: str, text_b: str) -> float:
    """Local cosine semantic similarity between two texts (0..1)."""
    return _cosine(_embed_vector(text_a), _embed_vector(text_b))


class Embedder(ABC):
    """Async embedding backend. Subclasses never raise on network problems.

    ``min_sim`` gates semantic recall: a doc with no lexical overlap must clear
    it to be returned. ``link_min_sim`` is the auto-link ``Related:`` threshold.
    """

    name: str = "embedder"
    min_sim: float = LOCAL_MIN_SIM
    link_min_sim: float = LOCAL_LINK_MIN_SIM
    available: bool = False

    @abstractmethod
    async def ensure_ready(self) -> bool:
        """Probe the backend so :attr:`available` is authoritative. Never raises."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Return a vector per text, or ``None`` if the backend is unavailable."""

    @staticmethod
    def similarity(a: Sequence[float], b: Sequence[float]) -> float:
        return _cosine(a, b)


class LocalEmbedder(Embedder):
    """Deterministic offline embedder (the historical hashing behaviour)."""

    name = "local-hash"
    available = True

    async def ensure_ready(self) -> bool:
        return True

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in _embed_vector(t)] for t in texts]


@dataclass(frozen=True)
class EmbeddingConfig:
    """Resolved ``[embeddings]`` section from ``~/.nelke/config.toml``."""

    base_url: str
    api_key: str | None = None
    #: Pinned model id; ``None`` means auto-detect an embedding model at runtime.
    model: str | None = None
    timeout: float = 10.0


def load_embedding_config(path: str | None = None) -> EmbeddingConfig | None:
    """Read the ``[embeddings]`` config section, or ``None`` when absent.

    Mirrors :func:`nelke.config.load_profiles` (``~/.nelke/config.toml``). A
    section without a usable ``base_url`` is treated as absent.
    """
    from nelke.config import default_nelke_home

    config_path = Path(path) if path else default_nelke_home() / "config.toml"
    try:
        with config_path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    cfg = data.get("embeddings") or {}
    if not isinstance(cfg, dict) or not str(cfg.get("base_url", "")).strip():
        return None
    model = str(cfg.get("model", "")).strip() or None
    return EmbeddingConfig(
        base_url=str(cfg["base_url"]).rstrip("/"),
        api_key=str(cfg["api_key"]) if cfg.get("api_key") else None,
        model=model,
        timeout=float(cfg.get("timeout", 10.0)),
    )


class OpenAICompatEmbedder(Embedder):
    """Dense embeddings via any OpenAI-compatible ``/embeddings`` endpoint.

    Works with LM Studio, Ollama, and cloud OpenAI-compatible providers. The
    model is either pinned via config or auto-detected from ``/v1/models`` by
    scanning for embedding-family name hints — so whichever (multilingual)
    embedding model the user has loaded in LM Studio is picked up automatically.
    """

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 10.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
        trust_env: bool = False,
        min_sim: float = 0.40,
        link_min_sim: float = 0.55,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "not-needed"
        self.pinned_model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_sim = min_sim
        self.link_min_sim = link_min_sim
        self._http_client: httpx.AsyncClient
        if http_client is not None:
            self._http_client = http_client
            self._own_client = False
        else:
            # httpx defaults to trust_env=True, which on Windows routes even
            # localhost through the OS proxy — same fix as llm.LLMClient.
            self._http_client = httpx.AsyncClient(timeout=timeout, trust_env=trust_env)
            self._own_client = True
        self._model = model
        self._cache: dict[str, list[float]] = {}
        self._ready: bool | None = None
        self._ready_at = 0.0

    @property
    def _embeddings_url(self) -> str:
        return f"{self.base_url}/embeddings"

    @property
    def _models_url(self) -> str:
        return f"{self.base_url}/models"

    async def close(self) -> None:
        if self._own_client:
            await self._http_client.aclose()

    async def ensure_ready(self) -> bool:
        """Resolve the model and mark :attr:`available`; honours a short TTL."""
        if self._ready is not None and time.monotonic() - self._ready_at < _DETECT_TTL:
            return self._ready
        try:
            if self._model is None:
                self._model = await self._detect_model()
            self._ready = bool(self._model)
        except Exception:  # noqa: BLE001 - backend probe must never raise
            self._ready = False
        self._ready_at = time.monotonic()
        self.available = bool(self._ready)
        return self._ready

    async def _detect_model(self) -> str | None:
        resp = await self._http_client.get(self._models_url, headers=self._headers())
        resp.raise_for_status()
        ids = [str(m.get("id", "")) for m in (resp.json().get("data") or [])]
        if ids:
            log.debug("embedding model candidates from %s: %s", self._models_url, ids)
        for mid in ids:
            low = mid.lower()
            if any(hint in low for hint in _EMBED_MODEL_HINTS):
                return mid
        return None

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not texts:
            return []
        if not await self.ensure_ready() or self._model is None:
            return None
        # Dedupe by text (cached vectors are reused); results reordered to match
        # the caller's original sequence.
        by_text: dict[str, list[float]] = {}
        order: list[str] = []
        missing: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            order.append(t)
            if t in by_text:
                continue
            cached = self._cache.get(t)
            if cached is not None:
                by_text[t] = cached
            else:
                missing.append((i, t))
        if missing:
            fetched = await self._request_embeddings([t for _, t in missing])
            if fetched is None:
                return None
            for (_, t), vec in zip(missing, fetched, strict=True):
                by_text[t] = vec
                self._cache[t] = vec
        return [by_text[t] for t in order]

    async def _request_embeddings(self, texts: list[str]) -> list[list[float]] | None:
        last: BaseException | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._http_client.post(
                    self._embeddings_url,
                    headers=self._headers(),
                    json={"model": self._model, "input": texts},
                )
                resp.raise_for_status()
                items = resp.json().get("data") or []
                # Sort by the "index" field when present (OpenAI spec); servers
                # like LM Studio/Ollama usually return input order.
                indexed = [
                    (int(item.get("index", idx)), [float(x) for x in item["embedding"]])
                    for idx, item in enumerate(items)
                    if isinstance(item, dict) and item.get("embedding")
                ]
                return [vec for _, vec in sorted(indexed, key=lambda pair: pair[0])]
            except Exception as exc:  # noqa: BLE001 - surfaced as unavailable
                last = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(2**attempt)
        log.warning("embeddings request failed after %s tries: %s", self.max_retries, last)
        return None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


_build_cache: dict[tuple[Any, ...], Embedder] = {}


def build_embedder(
    *,
    config: EmbeddingConfig | None = None,
    enabled: bool | None = None,
    path: str | None = None,
) -> Embedder:
    """Return the configured embedding backend, else the local one.

    ``enabled``: ``False`` forces the local embedder regardless of config; the
    default ``None`` consults ``Settings.embeddings_enabled`` (env
    ``NELKE_EMBEDDINGS_ENABLED``). ``config``/``path`` are test hooks. The
    result is cached per configuration so repeated stores share the model
    probing and connection.
    """
    from nelke.config import Settings

    if enabled is None:
        enabled = Settings().embeddings_enabled
    if not enabled:
        return LocalEmbedder()
    cfg = config if config is not None else load_embedding_config(path)
    if cfg is None:
        return LocalEmbedder()
    key: tuple[Any, ...] = ("emb", enabled, cfg.base_url, cfg.api_key, cfg.model, cfg.timeout)
    cached = _build_cache.get(key)
    if cached is not None:
        return cached
    embedder: Embedder = OpenAICompatEmbedder(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        timeout=cfg.timeout,
    )
    _build_cache[key] = embedder
    return embedder
