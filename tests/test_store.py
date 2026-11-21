"""Real embedded LanceDB tests; the deterministic embedder needs no model download."""

from datetime import UTC, datetime, timedelta

import pytest

from agent_memory.embedder import HashEmbedder
from agent_memory.store import MemoryStore


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
