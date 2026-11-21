"""Run the complete non-abstention LongMemEval retrieval split in resumable chunks.

Each child process releases its index and model before the next chunk starts.
The output directory is a checkpoint: a run with changed inputs must use a new directory.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

from evals.longmemeval_retrieval import load_cases


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_chunk(
    result: dict, question_ids: list[str], offset: int, k: int, modes: list[str]
) -> None:
    if result["question_ids"] != question_ids or result["offset"] != offset or result["k"] != k:
        raise ValueError(f"Chunk at offset {offset} has mismatched inputs")
    if set(result["results"]) != set(modes):
        raise ValueError(f"Chunk at offset {offset} has mismatched search modes")
    if result["indexed_sessions"] < len(question_ids):
        raise ValueError(f"Chunk at offset {offset} indexed too few sessions")
    for mode in modes:
        ranks = result["results"][mode]["ranks"]
        latencies = result["results"][mode]["latencies_ms"]
        if len(ranks) != len(question_ids) or len(latencies) != len(question_ids):
            raise ValueError(f"Chunk at offset {offset} has incomplete {mode} results")
        if any(not isinstance(rank, int) or rank < 0 or rank > k for rank in ranks):
            raise ValueError(f"Chunk at offset {offset} has invalid {mode} ranks")
        if any(
            not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0
            for latency in latencies
        ):
            raise ValueError(f"Chunk at offset {offset} has invalid {mode} latencies")


def summarize(chunks: list[dict], modes: list[str], k: int, case_count: int) -> dict:
    results = {}
    for mode in modes:
        ranks = [rank for chunk in chunks for rank in chunk["results"][mode]["ranks"]]
        latencies = [
            latency for chunk in chunks for latency in chunk["results"][mode]["latencies_ms"]
        ]
        if len(ranks) != case_count:
            raise ValueError(f"Expected {case_count} ranks for {mode}; got {len(ranks)}")
        results[mode] = {
            "hit_at_1": sum(rank == 1 for rank in ranks) / case_count,
            "hit_at_5": sum(0 < rank <= 5 for rank in ranks) / case_count,
            "hit_at_k": sum(rank > 0 for rank in ranks) / case_count,
            "mrr_at_k": sum(1 / rank for rank in ranks if rank) / case_count,
            "median_latency_ms": statistics.median(latencies),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--weights", nargs="+", type=float, default=[0.5, 0.7, 0.9])
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.k < 5 or any(not 0 <= weight <= 1 for weight in args.weights):
        parser.error("chunk-size must be positive, k >= 5, and weights must be in [0, 1]")
    if len(set(args.weights)) != len(args.weights):
        parser.error("weights must be unique")

    dataset = args.dataset.resolve(strict=True)
    cases = load_cases(dataset)
    question_ids = [case["question_id"] for case in cases]
    if not question_ids or len(set(question_ids)) != len(question_ids):
        raise ValueError("Dataset has no eligible questions or contains duplicate question IDs")
    adapter = Path(__file__).with_name("longmemeval_retrieval.py")
    manifest = {
        "dataset_sha256": sha256(dataset),
        "adapter_sha256": sha256(adapter),
        "runner_sha256": sha256(Path(__file__)),
        "question_count": len(question_ids),
        "question_ids": question_ids,
        "chunk_size": args.chunk_size,
        "k": args.k,
        "weights": args.weights,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                "Checkpoint manifest differs from this run; use a new output directory"
            )
    else:
        write_json(manifest_path, manifest)

    modes = ["vector:1.00", *(f"hybrid:{weight:.2f}" for weight in args.weights)]
    if len(set(modes)) != len(modes):
        parser.error("weights collapse to the same two-decimal mode key")
    chunks = []
    start = time.monotonic()
    for offset in range(0, len(cases), args.chunk_size):
        expected_ids = question_ids[offset : offset + args.chunk_size]
        chunk_path = args.output / f"chunk-{offset:04d}.json"
        if not chunk_path.exists():
            command = [
                sys.executable,
                "-u",
                str(adapter),
                "--dataset",
                str(dataset),
                "--offset",
                str(offset),
                "--limit",
                str(len(expected_ids)),
                "--json-output",
                str(chunk_path),
                "--k",
                str(args.k),
                "--weights",
                *(str(weight) for weight in args.weights),
                "--offline",
            ]
            print(f"Starting {offset + 1}-{offset + len(expected_ids)}/{len(cases)}", flush=True)
            subprocess.run(command, check=True)
        result = json.loads(chunk_path.read_text(encoding="utf-8"))
        validate_chunk(result, expected_ids, offset, args.k, modes)
        chunks.append(result)
        print(f"Completed {offset + len(expected_ids)}/{len(cases)}", flush=True)

    summary = {
        "status": "COMPLETE",
        "dataset_sha256": manifest["dataset_sha256"],
        "question_count": len(cases),
        "indexed_sessions": sum(chunk["indexed_sessions"] for chunk in chunks),
        "k": args.k,
        "chunk_size": args.chunk_size,
        "elapsed_seconds": time.monotonic() - start,
        "results": summarize(chunks, modes, args.k, len(cases)),
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
