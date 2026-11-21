"""An offline five-question retrieval contract with real LanceDB search."""

from agent_memory.embedder import HashEmbedder
from evals.run_eval import evaluate


def test_five_question_smoke_has_no_leak_and_hybrid_is_competitive():
    scores = evaluate(HashEmbedder(), limit=5)
    for mode in ("recency", "vector", "hybrid", "hybrid+history"):
        assert scores[(mode, "all")].leak_rate == 0
    assert scores[("hybrid", "all")].recall_5 >= scores[("vector", "all")].recall_5 - 0.2
