"""The deterministic model must never be selected accidentally."""

import pytest

from agent_memory.cli import main
from agent_memory.llm import FakeLLM, create_llm


def test_no_provider_requires_explicit_demo(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No live LLM configured"):
        create_llm()
    assert isinstance(create_llm(demo=True), FakeLLM)


def test_demo_refuses_provider_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Unset provider API keys"):
        create_llm(demo=True)


def test_cli_rejects_missing_provider_before_creating_database(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    database = tmp_path / "should-not-exist.lance"
    monkeypatch.setattr(
        "sys.argv", ["agent-memory", "chat", "--user", "alice", "--db", str(database)]
    )
    with pytest.raises(RuntimeError, match="No live LLM configured"):
        main()
    assert not database.exists()
