"""The public-benchmark adapter keeps each question's sessions isolated."""

import json

from agent_memory.embedder import HashEmbedder
from agent_memory.store import MemoryStore
from evals.longmemeval_retrieval import load_cases, measure, seed_sessions
from evals.run_longmemeval_full import summarize, validate_chunk


def test_longmemeval_adapter_scopes_sessions_and_skips_abstentions(tmp_path):
    case = {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "I live in Denver",
        "answer_session_ids": ["s1"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/20 (Sat) 02:57"],
        "haystack_sessions": [
            [{"role": "user", "content": "I live in Denver"}],
            [{"role": "user", "content": "I prefer tea"}],
        ],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case, dict(case, question_id="q2_abs")]), encoding="utf-8")
    cases = load_cases(path)
    assert len(cases) == 1
    assert load_cases(path, limit=1, offset=1) == []
    store = MemoryStore(tmp_path / "memory.lance", HashEmbedder())
    seed_sessions(store, cases, batch_size=1)
    assert store.table.count_rows() == 2
    assert all(row.user_id == "q1" for row in store.search("q1", "Denver", mode="hybrid"))
    assert measure(store, cases, "vector", 5)["hit_1"] == 1.0


def test_full_benchmark_aggregation_uses_question_counts_and_global_median():
    chunks = [
        {
            "question_ids": ["q1", "q2"],
            "offset": 0,
            "k": 10,
            "indexed_sessions": 4,
            "results": {
                "vector:1.00": {
                    "ranks": [1, 0],
                    "latencies_ms": [1.0, 10.0],
                }
            },
        },
        {
            "question_ids": ["q3"],
            "offset": 2,
            "k": 10,
            "indexed_sessions": 2,
            "results": {"vector:1.00": {"ranks": [5], "latencies_ms": [100.0]}},
        },
    ]
    modes = ["vector:1.00"]
    validate_chunk(chunks[0], ["q1", "q2"], 0, 10, modes)
    validate_chunk(chunks[1], ["q3"], 2, 10, modes)
    summary = summarize(chunks, modes, 10, 3)["vector:1.00"]
    assert summary["hit_at_1"] == 1 / 3
    assert summary["hit_at_5"] == 2 / 3
    assert summary["mrr_at_k"] == (1 + 1 / 5) / 3
    assert summary["median_latency_ms"] == 10
