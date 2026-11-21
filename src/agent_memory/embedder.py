"""Swappable, local text embeddings. The hash implementation is a test baseline."""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

DIMENSION = 384
MODEL_NAME = "BAAI/bge-small-en-v1.5"
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Embedding contract shared by the agent, store and evaluation."""

    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed memory text in insertion order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval question."""


class FastEmbedEmbedder:
    """On-device BGE embeddings. The public model is downloaded once, then cached."""

    dimension = DIMENSION

    def __init__(self, cache_dir: str = "data/models", *, local_files_only: bool = False) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.astype("float32").tolist() for vector in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(self._model.query_embed(text)).astype("float32").tolist()


class HashEmbedder:
    """Stable token-hash vectors for network-free tests; no semantic understanding."""

    dimension = DIMENSION

    def _embed(self, text: str) -> list[float]:
        counts: Counter[str] = Counter(_TOKEN.findall(text.lower()))
        vector = [0.0] * self.dimension
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
