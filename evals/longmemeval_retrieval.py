"""Session-retrieval pilot on the official LongMemEval cleaned small split.

This measures evidence-session retrieval, not answer accuracy. The official
dataset must be downloaded separately; it is never committed to this repo.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import NAMESPACE_URL, uuid5

import pyarrow as pa

from agent_memory.embedder import FastEmbedEmbedder
from agent_memory.store import MemoryStore


def load_cases(path: Path, limit: int | None = None) -> list[dict]:
    """Load the official JSON and omit abstentions, as its retrieval protocol does."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Expected a list of LongMemEval cases")
    cases = [case for case in cases if not case["question_id"].endswith("_abs")]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        cases = cases[:limit]
    return cases


def seed_sessions(store: MemoryStore, cases: list[dict], batch_size: int = 128) -> None:
    """Index one complete formatted conversation session per searchable row."""
    pending: list[dict] = []
    indexed = 0

    def flush() -> None:
        nonlocal indexed
        if not pending:
            return
        vectors = store.embedder.embed_documents([row["text"] for row in pending])
        for row, vector in zip(pending, vectors, strict=True):
            row["vector"] = vector
        store.table.add(pa.Table.from_pylist(pending, schema=store.table.schema))
        indexed += len(pending)
        print(f"Indexed {indexed} sessions", file=sys.stderr, flush=True)
        pending.clear()

    for case in cases:
        session_ids = case["haystack_session_ids"]
        sessions = case["haystack_sessions"]
        dates = case["haystack_dates"]
        if not len(session_ids) == len(sessions) == len(dates):
            raise ValueError(f"Session fields have different lengths: {case['question_id']}")
        for session_id, session, date in zip(session_ids, sessions, dates, strict=True):
            text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in session)
            if not text.strip():
                raise ValueError(f"Empty session: {case['question_id']}/{session_id}")
            pending.append(
                {
                    "id": str(
                        uuid5(NAMESPACE_URL, f"longmemeval/{case['question_id']}/{session_id}")
                    ),
                    "user_id": case["question_id"],
                    "text": text,
                    "kind": "event",
                    "topic": str(session_id),
                    "created_at": datetime.strptime(date, "%Y/%m/%d (%a) %H:%M").replace(
                        tzinfo=UTC
                    ),
                    "superseded": False,
                }
            )
            if len(pending) == batch_size:
                flush()
    flush()
    store.create_indexes()


def measure(store: MemoryStore, cases: list[dict], mode: str, k: int) -> dict[str, float]:
    """Measure evidence-session recall and MRR over non-abstention cases."""
    hits_1 = hits_5 = hits_k = 0
    reciprocal_rank = 0.0
    latencies = []
    for case in cases:
        start = time.perf_counter()
        found = store.search(case["question_id"], case["question"], k=k, mode=mode)
        latencies.append((time.perf_counter() - start) * 1000)
        expected = {
            str(uuid5(NAMESPACE_URL, f"longmemeval/{case['question_id']}/{session_id}"))
            for session_id in case["answer_session_ids"]
        }
        rank = next((i for i, row in enumerate(found, 1) if row.id in expected), None)
        hits_1 += rank == 1
        hits_5 += rank is not None and rank <= 5
        hits_k += rank is not None
        reciprocal_rank += 1 / rank if rank else 0
    n = len(cases)
    return {
        "recall_1": hits_1 / n,
        "recall_5": hits_5 / n,
        "recall_k": hits_k / n,
        "mrr": reciprocal_rank / n,
        "median_ms": statistics.median(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="First N non-abstention cases: pilot only")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.k < 5:
        parser.error("--k must be at least 5")
    cases = load_cases(args.dataset, args.limit)
    if not cases:
        parser.error("No non-abstention cases found")
    embedder = FastEmbedEmbedder(local_files_only=args.offline)
    with TemporaryDirectory(prefix="longmemeval-retrieval-") as temporary:
        store = MemoryStore(Path(temporary) / "memory.lance", embedder)
        seed_sessions(store, cases)
        print(f"Indexed {store.table.count_rows()} sessions for {len(cases)} questions")
        print("Mode,Weight,Recall@1,Recall@5,Recall@K,MRR,MedianMs")
        for mode, weight in [("vector", 1.0), *(("hybrid", w) for w in args.weights)]:
            store.hybrid_vector_weight = weight
            score = measure(store, cases, mode, args.k)
            print(
                f"{mode},{weight:.2f},{score['recall_1']:.3f},{score['recall_5']:.3f},"
                f"{score['recall_k']:.3f},{score['mrr']:.3f},{score['median_ms']:.1f}"
            )


if __name__ == "__main__":
    main()
