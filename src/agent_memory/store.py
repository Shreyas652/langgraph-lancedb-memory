"""Embedded LanceDB store with per-user search and topic supersession."""

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import lancedb
import pyarrow as pa
from lancedb.expr import col
from lancedb.index import FTS, BTree
from lancedb.query import ColumnOrdering
from lancedb.rerankers import RRFReranker

from agent_memory.embedder import Embedder

MemoryKind = Literal["fact", "preference", "event"]
SearchMode = Literal["vector", "hybrid", "recency"]


@dataclass(frozen=True, slots=True)
class Memory:
    id: str
    user_id: str
    text: str
    kind: str
    topic: str
    created_at: datetime
    superseded: bool

    @classmethod
    def from_row(cls, row: dict) -> "Memory":
        return cls(**{field: row[field] for field in cls.__dataclass_fields__})


class _WeightedRRFReranker(RRFReranker):
    """Fuse stable candidate ranks with an explicit vector/lexical balance."""

    def __init__(self, vector_weight: float) -> None:
        if not math.isfinite(vector_weight) or not 0 <= vector_weight <= 1:
            raise ValueError("hybrid_vector_weight must be between 0 and 1")
        super().__init__()
        self.vector_weight = vector_weight

    def rerank_hybrid(self, query, vector_results, fts_results):
        del query
        if vector_results.num_rows:
            vector_results = vector_results.sort_by(
                [("_distance", "ascending"), ("id", "ascending")]
            )
        if fts_results.num_rows:
            fts_results = fts_results.sort_by([("_score", "descending"), ("id", "ascending")])
        scores: defaultdict[int, float] = defaultdict(float)
        for results, weight in (
            (vector_results, self.vector_weight),
            (fts_results, 1 - self.vector_weight),
        ):
            for rank, row_id in enumerate(results["_rowid"].to_pylist(), 1):
                scores[row_id] += weight / (self.K + rank)
        fused = self.merge_results(vector_results, fts_results)
        fused = fused.append_column(
            "_relevance_score",
            pa.array([scores[row_id] for row_id in fused["_rowid"].to_pylist()], type=pa.float32()),
        )
        return fused.sort_by([("_relevance_score", "descending"), ("id", "ascending")])


class MemoryStore:
    """One local LanceDB table. Each read and mutation uses a typed user filter."""

    def __init__(
        self, uri: str | Path, embedder: Embedder, *, hybrid_vector_weight: float = 0.5
    ) -> None:
        if not math.isfinite(hybrid_vector_weight) or not 0 <= hybrid_vector_weight <= 1:
            raise ValueError("hybrid_vector_weight must be between 0 and 1")
        self.embedder = embedder
        self.hybrid_vector_weight = hybrid_vector_weight
        self.db = lancedb.connect(str(uri))
        schema = pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("user_id", pa.string(), nullable=False),
                pa.field("text", pa.string(), nullable=False),
                pa.field("kind", pa.string(), nullable=False),
                pa.field("topic", pa.string(), nullable=False),
                pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("superseded", pa.bool_(), nullable=False),
                pa.field("vector", pa.list_(pa.float32(), embedder.dimension), nullable=False),
            ]
        )
        if "memories" in self.db.list_tables().tables:
            self.table = self.db.open_table("memories")
            if self.table.schema != schema:
                raise ValueError("Existing memories table has an incompatible schema")
        else:
            self.table = self.db.create_table("memories", schema=schema)
        self._fts_dirty = True

    @staticmethod
    def _user_filter(user_id: str, include_superseded: bool):
        predicate = col("user_id") == user_id
        if not include_superseded:
            predicate = predicate & (col("superseded") == False)  # noqa: E712
        return predicate

    def add(
        self,
        user_id: str,
        text: str,
        kind: MemoryKind,
        topic: str,
        created_at: datetime | None = None,
        *,
        memory_id: str | None = None,
    ) -> str:
        """Add a memory; the newest active memory on this user's topic wins."""
        if not user_id.strip() or not text.strip() or not topic.strip():
            raise ValueError("user_id, text and topic must be nonempty")
        if kind not in ("fact", "preference", "event"):
            raise ValueError("kind must be fact, preference or event")
        timestamp = created_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        timestamp = timestamp.astimezone(UTC)
        vector = self.embedder.embed_documents([text])[0]
        if len(vector) != self.embedder.dimension:
            raise ValueError("Embedder returned the wrong vector dimension")

        active_filter = (
            (col("user_id") == user_id) & (col("topic") == topic) & (col("superseded") == False)  # noqa: E712
        )
        active = self.table.search().where(active_filter).to_list()
        is_historical = any(row["created_at"] > timestamp for row in active)
        memory_id = memory_id or str(uuid4())
        if self.table.search().where(col("id") == memory_id).limit(1).to_list():
            raise ValueError("memory_id already exists")
        new_row = {
            "id": memory_id,
            "user_id": user_id,
            "text": text,
            "kind": kind,
            "topic": topic,
            "created_at": timestamp,
            "superseded": is_historical,
            "vector": vector,
        }
        # A merge commits the old-row updates and new row together. A failed
        # write therefore leaves the previously active memory available.
        rows = [dict(row, superseded=True) for row in active] if not is_historical else []
        rows.append(new_row)
        merge = self.table.merge_insert("id").when_matched_update_all()
        merge.when_not_matched_insert_all().execute(
            pa.Table.from_pylist(rows, schema=self.table.schema)
        )
        self._fts_dirty = True
        return memory_id

    def create_indexes(self) -> None:
        """Build exact-search FTS and a user filter index; no vector ANN index."""
        if self.table.count_rows() == 0:
            return
        self.table.create_index("text", config=FTS(), replace=True)
        self.table.create_index("user_id", config=BTree(), replace=True)
        self._fts_dirty = False

    def search(
        self,
        user_id: str,
        query: str,
        k: int = 5,
        mode: SearchMode = "vector",
        include_superseded: bool = False,
    ) -> list[Memory]:
        """Return at most k memories from this user, with filters applied before ranking."""
        if not user_id.strip():
            raise ValueError("user_id must be nonempty")
        if not query.strip() and mode != "recency":
            raise ValueError("query must be nonempty for vector or hybrid search")
        if k < 1:
            raise ValueError("k must be positive")
        if self.table.count_rows() == 0:
            return []
        predicate = self._user_filter(user_id, include_superseded)
        if mode == "recency":
            builder = (
                self.table.search()
                .where(predicate)
                .order_by(
                    [
                        ColumnOrdering(column_name="created_at", ascending=False),
                        ColumnOrdering(column_name="id", ascending=False),
                    ]
                )
            )
        elif mode == "vector":
            vector = self.embedder.embed_query(query)
            builder = self.table.search(vector, query_type="vector").where(
                predicate, prefilter=True
            )
        elif mode == "hybrid":
            if self._fts_dirty:
                self.create_indexes()
            vector = self.embedder.embed_query(query)
            builder = (
                self.table.search(query_type="hybrid")
                .vector(vector)
                .text(query)
                .where(predicate, prefilter=True)
                .rerank(_WeightedRRFReranker(self.hybrid_vector_weight))
            )
        else:
            raise ValueError(f"Unknown search mode: {mode}")
        # Fuse a bounded candidate pool before applying the requested result
        # limit. Otherwise tied FTS hits can be discarded at the k boundary
        # before the deterministic RRF tie-break has a chance to order them.
        candidate_limit = k * 4 if mode == "hybrid" else k
        return [Memory.from_row(row) for row in builder.limit(candidate_limit).to_list()[:k]]

    def delete_user(self, user_id: str) -> None:
        """Remove all of one user's memories, including superseded history."""
        if not user_id.strip():
            raise ValueError("user_id must be nonempty")
        self.table.delete(col("user_id") == user_id)
        self._fts_dirty = True
