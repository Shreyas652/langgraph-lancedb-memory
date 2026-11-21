"""The public-benchmark adapter keeps each question's sessions isolated."""

import json

from agent_memory.embedder import HashEmbedder
from agent_memory.store import MemoryStore
from evals.longmemeval_retrieval import load_cases, measure, seed_sessions


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
    store = MemoryStore(tmp_path / "memory.lance", HashEmbedder())
    seed_sessions(store, cases, batch_size=1)
    assert store.table.count_rows() == 2
    assert all(row.user_id == "q1" for row in store.search("q1", "Denver", mode="hybrid"))
    assert measure(store, cases, "vector", 5)["recall_1"] == 1.0
