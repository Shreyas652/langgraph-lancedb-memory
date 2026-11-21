"""Compare exact vector scan with an IVF_FLAT index on identical memory rows.

Synthetic vectors isolate storage/query scaling; they are not an embedding
quality benchmark. The JSON records parameters, timings, and ANN recall
against exact search so results can be independently checked.
"""

import argparse
import json
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
from lancedb.index import IvfFlat

from agent_memory.embedder import DIMENSION, HashEmbedder
from agent_memory.store import MemoryStore


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (
        index - lower
    )


def run(
    workdir: Path,
    *,
    rows: int,
    queries: int,
    batch_size: int,
    partitions: int,
    nprobes: int,
    seed: int,
) -> dict:
    if min(rows, queries, batch_size, partitions, nprobes) < 1:
        raise ValueError("rows, queries, batch_size, partitions and nprobes must be positive")
    if partitions > rows or nprobes > partitions:
        raise ValueError("Require nprobes <= partitions <= rows")
    workdir.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(workdir / "memory.lance", HashEmbedder())
    if store.table.count_rows():
        raise ValueError("Use an empty workdir; existing rows would invalidate the benchmark")
    rng = np.random.default_rng(seed)
    selected = rng.choice(rows, size=queries, replace=True)
    query_vectors: dict[int, np.ndarray] = {}
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    start = time.perf_counter()
    for offset in range(0, rows, batch_size):
        count = min(batch_size, rows - offset)
        vectors = rng.normal(size=(count, DIMENSION)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        for wanted in selected:
            if offset <= wanted < offset + count:
                query_vectors[int(wanted)] = vectors[wanted - offset].copy()
        batch = [
            {
                "id": f"memory-{i:09d}",
                "user_id": "scale-user",
                "text": f"Synthetic memory {i}",
                "kind": "fact",
                "topic": f"topic-{i}",
                "created_at": timestamp,
                "superseded": False,
                "vector": vector.tolist(),
            }
            for i, vector in enumerate(vectors, offset)
        ]
        store.table.add(pa.Table.from_pylist(batch, schema=store.table.schema))
    ingest_seconds = time.perf_counter() - start
    if store.table.count_rows() != rows or len(query_vectors) != len(set(selected)):
        raise AssertionError("Ingestion did not preserve every expected row and query vector")
    vectors = [query_vectors[int(i)].tolist() for i in selected]

    def exact(vector: list[float]) -> list[str]:
        return [
            row["id"]
            for row in store.table.search(vector, query_type="vector")
            .bypass_vector_index()
            .limit(10)
            .to_list()
        ]

    def ann(vector: list[float]) -> list[str]:
        return [
            row["id"]
            for row in store.table.search(vector, query_type="vector")
            .nprobes(nprobes)
            .limit(10)
            .to_list()
        ]

    # One warm query avoids comparing an ANN warm start with a cold exact scan.
    exact(vectors[0])
    exact_times = []
    gold = []
    for vector in vectors:
        start = time.perf_counter()
        gold.append(exact(vector))
        exact_times.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    store.table.create_index("vector", config=IvfFlat(num_partitions=partitions))
    build_seconds = time.perf_counter() - start
    index_info = store.table.list_indices()
    if not any("vector" in item.columns for item in index_info):
        raise AssertionError("Vector index was not created")
    ann(vectors[0])
    ann_times = []
    found = []
    for vector in vectors:
        start = time.perf_counter()
        found.append(ann(vector))
        ann_times.append((time.perf_counter() - start) * 1000)
    exact_after_times = []
    for vector in vectors:
        start = time.perf_counter()
        exact(vector)
        exact_after_times.append((time.perf_counter() - start) * 1000)
    bytes_on_disk = sum(path.stat().st_size for path in workdir.rglob("*") if path.is_file())
    return {
        "status": "COMPLETE",
        "rows": rows,
        "dimension": DIMENSION,
        "queries": queries,
        "seed": seed,
        "index": "IVF_FLAT",
        "partitions": partitions,
        "nprobes": nprobes,
        "ingest_seconds": ingest_seconds,
        "index_build_seconds": build_seconds,
        "bytes_on_disk": bytes_on_disk,
        "exact_before_p50_ms": statistics.median(exact_times),
        "exact_before_p95_ms": percentile(exact_times, 0.95),
        "exact_after_p50_ms": statistics.median(exact_after_times),
        "exact_after_p95_ms": percentile(exact_after_times, 0.95),
        "indexed_p50_ms": statistics.median(ann_times),
        "indexed_p95_ms": percentile(ann_times, 0.95),
        "indexed_recall_at_10": sum(
            len(set(expected) & set(actual)) / len(expected)
            for expected, actual in zip(gold, found, strict=True)
        )
        / queries,
        "indexed_top_1_agreement": sum(
            expected[0] == actual[0] for expected, actual in zip(gold, found, strict=True)
        )
        / queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--partitions", type=int, default=256)
    parser.add_argument("--nprobes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="memory-scale-") as temporary:
        result = run(
            Path(temporary),
            rows=args.rows,
            queries=args.queries,
            batch_size=args.batch_size,
            partitions=args.partitions,
            nprobes=args.nprobes,
            seed=args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary_output.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
