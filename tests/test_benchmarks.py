"""Benchmark protocol checks that do not call a live model."""

from datetime import UTC, datetime

import pytest

from agent_memory.store import Memory
from evals.longmemeval_answers import build_context, parse_verdict, summarize
from evals.scale_100k import run


def test_context_uses_only_complete_retrieved_sessions():
    case = {
        "question_id": "q",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2026/01/01 (Thu) 00:00", "2026/01/02 (Fri) 00:00"],
    }
    memories = [
        Memory(str(i), "q", text, "event", topic, datetime(2026, 1, 1, tzinfo=UTC), False)
        for i, (text, topic) in enumerate([("first", "s1"), ("second", "s2")])
    ]
    context, chosen = build_context(case, memories, 52)
    assert chosen == ["s1"]
    assert "first" in context and "second" not in context
    with pytest.raises(ValueError, match="No complete retrieved session"):
        build_context(case, memories, 1)


def test_judge_verdict_and_paired_summary():
    assert parse_verdict("YES")
    assert not parse_verdict("No. Explanation")
    with pytest.raises(ValueError, match="YES or NO"):
        parse_verdict("Maybe")
    records = [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "baseline_correct": False,
            "memory_correct": True,
            "gold_session_ids": ["s1", "s2"],
            "retrieved_session_ids": ["s1", "s2"],
            "context_session_ids": ["s1", "s2"],
            "usage": {"reader": {"input_tokens": 5, "output_tokens": 2}},
        },
        {
            "question_id": "q2_abs",
            "question_type": "single-session-user",
            "baseline_correct": True,
            "memory_correct": True,
            "gold_session_ids": [],
            "retrieved_session_ids": [],
            "context_session_ids": [],
            "usage": {"reader": {"input_tokens": 3, "output_tokens": 1}},
        },
    ]
    result = summarize(records, 2)
    assert result["baseline_accuracy"] == 0.5
    assert result["memory_accuracy"] == 1
    assert result["strict_all_evidence_at_context"] == 1
    assert result["paired_memory_only_wins"] == 1
    assert result["token_usage"]["reader_input"] == 8


def test_index_benchmark_uses_same_rows_for_exact_and_ann(tmp_path):
    result = run(
        tmp_path / "scale",
        rows=1_000,
        queries=5,
        batch_size=250,
        partitions=8,
        nprobes=8,
        seed=1,
    )
    assert result["status"] == "COMPLETE"
    assert result["rows"] == 1_000
    assert result["indexed_recall_at_10"] == 1
