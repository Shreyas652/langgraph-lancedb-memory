"""Recompute LongMemEval answer scores from committed, text-free labels."""

import argparse
import hashlib
import json
import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _equal(actual: float, reported: float) -> None:
    if not math.isclose(actual, reported, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Metric mismatch: computed {actual}, reported {reported}")


def verify(
    manifest_path: Path,
    summary_path: Path,
    labels_path: Path,
    *,
    dataset_path: Path | None = None,
    source_archive: Path | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = _read_jsonl(labels_path)
    expected_ids = manifest["question_ids"]
    actual_ids = [row["question_id"] for row in records]
    if (
        summary["status"] != "COMPLETE"
        or len(expected_ids) != len(set(expected_ids))
        or len(actual_ids) != len(set(actual_ids))
        or set(actual_ids) != set(expected_ids)
        or len(records) != summary["question_count"]
    ):
        raise ValueError("Incomplete, duplicate, or mismatched question coverage")
    if manifest["dataset_sha256"] != summary["dataset_sha256"]:
        raise ValueError("Dataset hash differs between manifest and summary")
    for field in ("reader_model", "judge_model", "k", "max_context_chars"):
        if manifest[field] != summary[field]:
            raise ValueError(f"Run configuration differs: {field}")
    if dataset_path is not None:
        if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
            raise ValueError("Dataset file hash differs from the run manifest")
        cases = json.loads(dataset_path.read_text(encoding="utf-8"))
        if [case["question_id"] for case in cases] != expected_ids:
            raise ValueError("Dataset questions differ from the run manifest")
        cases_by_id = {case["question_id"]: case for case in cases}
    else:
        cases_by_id = None
    if source_archive is not None:
        with tarfile.open(source_archive) as archive:
            source = archive.extractfile("evals/longmemeval_answers.py")
            if source is None:
                raise ValueError("Source archive omits the answer runner")
            runner_hash = hashlib.sha256(source.read()).hexdigest()
        if runner_hash != manifest["runner_sha256"]:
            raise ValueError("Source archive runner hash differs from manifest")

    counts: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    tokens: Counter[str] = Counter()
    for row in records:
        question_id = row["question_id"]
        if cases_by_id is not None:
            case = cases_by_id[question_id]
            if (
                row["question_type"] != case["question_type"]
                or row["gold_session_ids"] != case["answer_session_ids"]
                or not set(row["retrieved_session_ids"]) <= set(case["haystack_session_ids"])
            ):
                raise ValueError(f"Dataset labels or retrieval IDs differ: {question_id}")
        retrieved = row["retrieved_session_ids"]
        context = row["context_session_ids"]
        if (
            len(retrieved) > manifest["k"]
            or len(retrieved) != len(set(retrieved))
            or len(context) != len(set(context))
            or not set(context) <= set(retrieved)
        ):
            raise ValueError(f"Invalid retrieved or context IDs: {question_id}")
        for arm in ("baseline", "memory"):
            label = row[f"{arm}_correct"]
            if not isinstance(label, bool):
                raise ValueError(f"Non-boolean verdict: {question_id}/{arm}")
            counts[f"{arm}_correct"] += label
            by_type[row["question_type"]][f"{arm}_correct"] += label
            for role in ("reader", "judge"):
                for kind in ("input", "output"):
                    value = row["usage"][f"{arm}_{role}"][f"{kind}_tokens"]
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        raise ValueError(f"Invalid token usage: {question_id}/{arm}_{role}")
                    tokens[f"{arm}_{role}_{kind}"] += value
        counts["memory_only_wins"] += row["memory_correct"] and not row["baseline_correct"]
        counts["baseline_only_wins"] += row["baseline_correct"] and not row["memory_correct"]
        by_type[row["question_type"]]["n"] += 1
        if not question_id.endswith("_abs"):
            counts["nonabstention"] += 1
            gold = set(row["gold_session_ids"])
            for scope, found in (("k", retrieved), ("context", context)):
                counts[f"all_{scope}"] += gold <= set(found)
                counts[f"any_{scope}"] += bool(gold & set(found))

    n = len(records)
    _equal(counts["baseline_correct"] / n, summary["baseline_accuracy"])
    _equal(counts["memory_correct"] / n, summary["memory_accuracy"])
    if (
        counts["memory_only_wins"] != summary["paired_memory_only_wins"]
        or counts["baseline_only_wins"] != summary["paired_baseline_only_wins"]
        or dict(tokens) != summary["token_usage"]
    ):
        raise ValueError("Paired wins, losses, or token totals differ from summary")
    for scope in ("k", "context"):
        _equal(
            counts[f"all_{scope}"] / counts["nonabstention"],
            summary[f"strict_all_evidence_at_{scope}"],
        )
        _equal(
            counts[f"any_{scope}"] / counts["nonabstention"],
            summary[f"any_evidence_at_{scope}"],
        )
    if set(by_type) != set(summary["by_type"]):
        raise ValueError("Question-type coverage differs from summary")
    for question_type, values in by_type.items():
        reported = summary["by_type"][question_type]
        if values["n"] != reported["n"]:
            raise ValueError(f"Question-type count differs: {question_type}")
        for arm in ("baseline", "memory"):
            _equal(values[f"{arm}_correct"] / values["n"], reported[f"{arm}_accuracy"])
    return {"status": "VERIFIED", "question_count": n, "counts": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parent
    parser.add_argument(
        "--manifest", type=Path, default=directory / "longmemeval_answer_manifest.json"
    )
    parser.add_argument(
        "--summary", type=Path, default=directory / "longmemeval_answer_summary.json"
    )
    parser.add_argument(
        "--labels", type=Path, default=directory / "longmemeval_answer_labels.jsonl"
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.manifest,
                args.summary,
                args.labels,
                dataset_path=args.dataset,
                source_archive=args.source_archive,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
