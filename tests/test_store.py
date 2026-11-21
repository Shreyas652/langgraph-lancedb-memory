"""Real embedded LanceDB tests; the deterministic embedder needs no model download."""

import json
import sys
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from agent_memory.cli import main
from agent_memory.embedder import HashEmbedder
from agent_memory.store import MemoryStore, _WeightedRRFReranker


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "memory.lance", HashEmbedder())


def test_add_search_and_supersede_by_user_and_topic(store):
    first = datetime(2026, 1, 1, tzinfo=UTC)
    old = store.add("alice", "I live in Austin", "fact", "home_city", first)
    current = store.add(
        "alice", "I moved to Denver", "fact", "home_city", first + timedelta(days=1)
    )
    bob = store.add("bob", "I live in Austin", "fact", "home_city", first)
    store.create_indexes()

    for mode in ("recency", "vector", "hybrid"):
        found = store.search("alice", "Where does Alice live?", mode=mode)
        assert [memory.id for memory in found] == [current]
        assert all(memory.user_id == "alice" for memory in found)
    history = store.search("alice", "city", mode="recency", include_superseded=True)
    assert {memory.id for memory in history} == {old, current}
    assert next(memory for memory in history if memory.id == old).superseded
    assert store.search("bob", "city", mode="recency")[0].id == bob


def test_quotes_in_user_ids_are_typed_data_not_sql(store):
    dangerous = "a' OR user_id != 'a"
    own = store.add(dangerous, "Prefers tea", "preference", "drink")
    store.add("victim", "Prefers coffee", "preference", "drink")
    assert [memory.id for memory in store.search(dangerous, "tea", mode="hybrid")] == [own]
    store.delete_user(dangerous)
    assert store.search(dangerous, "tea", mode="recency") == []
    assert len(store.search("victim", "coffee", mode="recency")) == 1


def test_historical_insert_does_not_replace_newer_memory(store):
    newer = datetime(2026, 2, 1, tzinfo=UTC)
    active = store.add("alice", "Lives in Denver", "fact", "home_city", newer)
    store.add("alice", "Lived in Austin", "fact", "home_city", newer - timedelta(days=1))
    assert [memory.id for memory in store.search("alice", "city", mode="recency")] == [active]


def test_duplicate_fact_is_merged_and_timeline_answers_before_after(store):
    first = datetime(2026, 1, 1, tzinfo=UTC)
    second = first + timedelta(days=10)
    austin = store.add("alice", "Lives in Austin", "fact", "home_city", first)
    assert store.add("alice", "Lives in Austin", "fact", "home_city", first) == austin
    assert (
        store.add("alice", "Lives in Austin", "fact", "home_city", first + timedelta(days=1))
        == austin
    )
    denver = store.add("alice", "Lives in Denver", "fact", "home_city", second)
    assert store.add("alice", "Lives in Denver", "fact", "home_city", second) == denver
    assert store.table.count_rows() == 2
    before, after = store.compare_topic(
        "alice", "home_city", first + timedelta(days=1), second + timedelta(days=1)
    )
    assert (before.id, after.id) == (austin, denver)
    assert store.get_topic_at("alice", "home_city", first - timedelta(days=1)) is None
    assert store.get_topic_at("bob", "home_city", second + timedelta(days=1)) is None


def test_temporal_backfill_and_conflicting_timestamp(store):
    first = datetime(2026, 1, 1, tzinfo=UTC)
    third = first + timedelta(days=20)
    store.add("alice", "Lives in Denver", "fact", "home_city", third)
    austin = store.add("alice", "Lives in Austin", "fact", "home_city", first)
    assert store.add("alice", "Lives in Austin", "fact", "home_city", first) == austin
    assert store.get_topic_at("alice", "home_city", first + timedelta(days=1)).id == austin
    with pytest.raises(ValueError, match="same timestamp"):
        store.add("alice", "Lives in Phoenix", "fact", "home_city", third)
    with pytest.raises(ValueError, match="same timestamp"):
        store.add("alice", "Lives in Phoenix", "fact", "home_city", first)
    with pytest.raises(ValueError, match="timezone"):
        store.get_topic_at("alice", "home_city", datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="before must precede after"):
        store.compare_topic("alice", "home_city", third, first)


def test_compare_cli_returns_before_and_after_without_model(tmp_path, monkeypatch, capsys):
    path = tmp_path / "memory.lance"
    store = MemoryStore(path, HashEmbedder())
    store.add("alice", "Austin", "fact", "home_city", datetime(2026, 1, 1, tzinfo=UTC))
    store.add("alice", "Denver", "fact", "home_city", datetime(2026, 2, 1, tzinfo=UTC))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent-memory",
            "compare",
            "--user",
            "alice",
            "--topic",
            "home_city",
            "--before",
            "2026-01-15T00:00:00+00:00",
            "--after",
            "2026-02-15T00:00:00+00:00",
            "--db",
            str(path),
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out) == {"before": "Austin", "after": "Denver"}


def test_duplicate_id_cannot_replace_another_users_memory(store):
    existing = store.add("bob", "Prefers coffee", "preference", "drink")
    old = store.add("alice", "Lives in Austin", "fact", "home_city")
    with pytest.raises(ValueError, match="memory_id already exists"):
        store.add(
            "alice",
            "Lives in Denver",
            "fact",
            "home_city",
            memory_id=existing,
        )
    assert [memory.id for memory in store.search("alice", "city", mode="recency")] == [old]
    assert [memory.id for memory in store.search("bob", "coffee", mode="recency")] == [existing]


def test_input_validation_and_schema_mismatch(store, tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        store.add("", "A fact", "fact", "topic")
    with pytest.raises(ValueError, match="timezone"):
        store.add("alice", "A fact", "fact", "topic", datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="positive"):
        store.search("alice", "fact", k=0)

    class SmallEmbedder(HashEmbedder):
        dimension = 8

    with pytest.raises(ValueError, match="incompatible"):
        MemoryStore(tmp_path / "memory.lance", SmallEmbedder())
    with pytest.raises(ValueError, match="hybrid_vector_weight"):
        MemoryStore(tmp_path / "memory.lance", HashEmbedder(), hybrid_vector_weight=float("nan"))


def test_weighted_fusion_changes_rank_when_search_legs_disagree():
    vector = pa.table({"id": ["A", "B"], "_rowid": [1, 2], "_distance": [0.1, 0.2]})
    lexical = pa.table({"id": ["B", "A"], "_rowid": [2, 1], "_score": [5.0, 1.0]})
    vector_first = _WeightedRRFReranker(0.9).rerank_hybrid("query", vector, lexical)
    lexical_first = _WeightedRRFReranker(0.1).rerank_hybrid("query", vector, lexical)
    assert vector_first["id"].to_pylist() == ["A", "B"]
    assert lexical_first["id"].to_pylist() == ["B", "A"]
