"""Measure local memory retrieval with real LanceDB searches and synthetic labels."""

import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import NAMESPACE_URL, uuid5

from agent_memory.embedder import Embedder, FastEmbedEmbedder, HashEmbedder
from agent_memory.store import Memory, MemoryStore

DATASET = Path(__file__).with_name("dataset.jsonl")
RESULTS = Path(__file__).with_name("results.md")
MODES = ("recency", "vector", "hybrid", "hybrid+history")
QUESTION_TYPES = ("direct", "paraphrase", "updated", "cross_user")


@dataclass(frozen=True)
class Observation:
    mode: str
    question_type: str
    ranks: tuple[str, ...]
    gold: frozenset[str]
    stale: frozenset[str]
    leaked: bool
    latency_ms: float


@dataclass(frozen=True)
class Score:
    count: int
    recall_1: float
    recall_3: float
    recall_5: float
    mrr: float
    stale_rate: float | None
    leak_rate: float
    median_latency_ms: float


def load_dataset(path: Path = DATASET) -> tuple[list[dict], list[dict]]:
    """Read and validate the hand-labeled synthetic memories and questions."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    memories = [row for row in rows if row.get("record") == "memory"]
    questions = [row for row in rows if row.get("record") == "question"]
    if len(memories) + len(questions) != len(rows):
        raise ValueError("Unknown dataset record type")
    by_id = {row["id"]: row for row in memories}
    if len(by_id) != len(memories) or len({row["id"] for row in questions}) != len(questions):
        raise ValueError("Duplicate dataset id")
    for question in questions:
        if question["type"] not in QUESTION_TYPES or not question["gold_ids"]:
            raise ValueError(f"Invalid question: {question['id']}")
        for memory_id in question["gold_ids"] + question.get("stale_ids", []):
            if memory_id not in by_id or by_id[memory_id]["user_id"] != question["user_id"]:
                raise ValueError(f"Gold/stale memory belongs to another user: {question['id']}")
        if question["type"] == "updated" and not question.get("stale_ids"):
            raise ValueError(f"Updated question needs stale IDs: {question['id']}")
    return memories, questions


def _score(observations: list[Observation]) -> Score:
    count = len(observations)

    def first_gold(item: Observation) -> int | None:
        return next(
            (index for index, memory_id in enumerate(item.ranks, 1) if memory_id in item.gold), None
        )

    def recall_at(k: int) -> float:
        return (
            sum((rank := first_gold(item)) is not None and rank <= k for item in observations)
            / count
        )

    updated = [item for item in observations if item.question_type == "updated"]
    stale_rate = None
    if updated:
        stale_above = 0
        for item in updated:
            gold_position = first_gold(item) or float("inf")
            if any(
                position < gold_position
                for position, memory_id in enumerate(item.ranks, 1)
                if memory_id in item.stale
            ):
                stale_above += 1
        stale_rate = stale_above / len(updated)
    return Score(
        count=count,
        recall_1=recall_at(1),
        recall_3=recall_at(3),
        recall_5=recall_at(5),
        mrr=sum(1 / rank if (rank := first_gold(item)) else 0 for item in observations) / count,
        stale_rate=stale_rate,
        leak_rate=sum(item.leaked for item in observations) / count,
        median_latency_ms=statistics.median(item.latency_ms for item in observations),
    )


def evaluate(
    embedder: Embedder,
    *,
    dataset: Path = DATASET,
    limit: int | None = None,
) -> dict[tuple[str, str], Score]:
    """Seed an isolated store, run each query mode, and return measured scores."""
    memories, questions = load_dataset(dataset)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        questions = questions[:limit]
    with TemporaryDirectory(prefix="agent-memory-eval-") as temporary:
        store = MemoryStore(Path(temporary) / "memory.lance", embedder)
        ids: dict[str, str] = {}
        for memory in sorted(memories, key=lambda row: (row["created_at"], row["id"])):
            ids[memory["id"]] = store.add(
                memory["user_id"],
                memory["text"],
                memory["kind"],
                memory["topic"],
                datetime.fromisoformat(memory["created_at"].replace("Z", "+00:00")),
                memory_id=str(uuid5(NAMESPACE_URL, f"langgraph-lancedb-memory/{memory['id']}")),
            )
        store.create_indexes()
        by_real_id = {real_id: logical_id for logical_id, real_id in ids.items()}
        observations: list[Observation] = []
        for mode in MODES:
            search_mode = "hybrid" if mode == "hybrid+history" else mode
            for question in questions:
                started = time.perf_counter()
                found: list[Memory] = store.search(
                    question["user_id"],
                    question["question"],
                    k=5,
                    mode=search_mode,
                    include_superseded=(mode == "hybrid+history"),
                )
                observations.append(
                    Observation(
                        mode=mode,
                        question_type=question["type"],
                        ranks=tuple(by_real_id[memory.id] for memory in found),
                        gold=frozenset(question["gold_ids"]),
                        stale=frozenset(question.get("stale_ids", [])),
                        leaked=any(memory.user_id != question["user_id"] for memory in found),
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )
                )
    grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for item in observations:
        grouped[(item.mode, "all")].append(item)
        grouped[(item.mode, item.question_type)].append(item)
    return {key: _score(items) for key, items in grouped.items()}


def format_table(scores: dict[tuple[str, str], Score]) -> str:
    """Render one comparison table with overall and question-type rows."""
    lines = [
        "| Mode | Type | N | Recall@1 | Recall@3 | Recall@5 | MRR | "
        "Stale rate | Leak rate | Median ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        for question_type in ("all", *QUESTION_TYPES):
            if (mode, question_type) not in scores:
                continue
            score = scores[(mode, question_type)]
            stale = "n/a" if score.stale_rate is None else f"{score.stale_rate:.1%}"
            lines.append(
                f"| {mode} | {question_type} | {score.count} | {score.recall_1:.1%} | "
                f"{score.recall_3:.1%} | {score.recall_5:.1%} | {score.mrr:.3f} | "
                f"{stale} | {score.leak_rate:.1%} | {score.median_latency_ms:.1f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedder", choices=("fastembed", "hash"), default="fastembed")
    parser.add_argument("--offline", action="store_true", help="Require the cached BGE model")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    embedder = (
        FastEmbedEmbedder(local_files_only=args.offline)
        if args.embedder == "fastembed"
        else HashEmbedder()
    )
    scores = evaluate(embedder, limit=args.limit)
    table = format_table(scores)
    print(table)
    if args.limit is None:
        report = (
            "# Retrieval evaluation\n\n"
            "Synthetic, hand-written data: 54 memories, three users, four sessions per user, "
            "45 questions. The measured embedder is "
            + (
                "FastEmbed BAAI/bge-small-en-v1.5"
                if args.embedder == "fastembed"
                else "HashEmbedder"
            )
            + ". Each search is filtered to the question's user before ranking. "
            "Recall@k means at least one gold memory occurs in the first k results. "
            "MRR uses the first gold rank. Stale rate is measured only on updated questions "
            "and means a superseded memory appeared ahead of the current answer in the top five. "
            "Leak rate means any returned memory belonged to a different user. "
            "Latency includes embedding and local search; it varies by machine.\n\n" + table + "\n"
        )
        args.output.write_text(report, encoding="utf-8")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
