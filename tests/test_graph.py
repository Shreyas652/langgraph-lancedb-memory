"""Short-term checkpoints and long-term memory are separate by design."""

from langchain_core.messages import HumanMessage

from agent_memory.embedder import HashEmbedder
from agent_memory.graph import build_graph
from agent_memory.llm import FakeLLM
from agent_memory.store import MemoryStore


class RecordingLLM:
    def __init__(self):
        self.turns = []

    def respond(self, system_prompt, messages, recalled):
        del system_prompt, recalled
        self.turns.append([(message.type, message.content) for message in messages])
        return "acknowledged"

    def extract(self, user_text):
        del user_text
        return []


def test_second_turn_passes_thread_history_to_model(tmp_path):
    store = MemoryStore(tmp_path / "memory.lance", HashEmbedder())
    llm = RecordingLLM()
    graph = build_graph(store, llm)
    config = {"configurable": {"thread_id": "alice:one"}}
    graph.invoke({"user_id": "alice", "messages": [HumanMessage("Hello")]}, config)
    graph.invoke({"user_id": "alice", "messages": [HumanMessage("What did I say?")]}, config)
    assert llm.turns[1] == [
        ("human", "Hello"),
        ("ai", "acknowledged"),
        ("human", "What did I say?"),
    ]


def test_new_thread_recalls_memory_from_previous_thread(tmp_path):
    store = MemoryStore(tmp_path / "memory.lance", HashEmbedder())
    graph = build_graph(store, FakeLLM())
    first = graph.invoke(
        {"user_id": "alice", "messages": [HumanMessage("I live in Denver.")]},
        {"configurable": {"thread_id": "alice:one"}},
    )
    assert first["new_memories"][0]["topic"] == "home_city"
    second = graph.invoke(
        {"user_id": "alice", "messages": [HumanMessage("Where do I live?")]},
        {"configurable": {"thread_id": "alice:two"}},
    )
    assert "I live in Denver" in second["messages"][-1].content
    assert second["new_memories"] == []


def test_checkpoint_thread_cannot_be_reused_by_other_user(tmp_path):
    store = MemoryStore(tmp_path / "memory.lance", HashEmbedder())
    graph = build_graph(store, FakeLLM())
    graph.invoke(
        {"user_id": "alice", "messages": [HumanMessage("I prefer tea.")]},
        {"configurable": {"thread_id": "shared"}},
    )
    try:
        graph.invoke(
            {"user_id": "bob", "messages": [HumanMessage("What do you remember?")]},
            {"configurable": {"thread_id": "shared"}},
        )
    except ValueError as exc:
        assert "another user" in str(exc)
    else:
        raise AssertionError("cross-user thread reuse was allowed")
