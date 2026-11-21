"""Independently check published text-free GPT-4o labels and aggregates."""

import json
import math
from collections import defaultdict
from pathlib import Path

from evals.official_gpt4o import MODEL, _jsonl


def verify(manifest_path: Path, labels_path: Path, summary_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _jsonl(labels_path)
    ids = [row["question_id"] for row in rows]
    if ids != manifest["question_ids"] or len(ids) != len(set(ids)):
        raise ValueError("published labels do not match manifest IDs")
    if len(rows) != manifest["questions"] or summary["questions"] != len(rows):
        raise ValueError("published question count differs")
    if manifest["grader_model"] != MODEL or summary["grader_model"] != MODEL:
        raise ValueError("published grader model differs")
    if not all(
        set(row) == {"question_id", "question_type", "baseline_correct", "memory_correct"}
        and isinstance(row["question_type"], str)
        and row["question_type"]
        and type(row["baseline_correct"]) is bool
        and type(row["memory_correct"]) is bool
        for row in rows
    ):
        raise ValueError("published label has an invalid type")

    for arm in ("baseline", "memory"):
        grouped: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            grouped[row["question_type"]].append(row[f"{arm}_correct"])
        actual = summary["arms"][arm]
        correct = sum(row[f"{arm}_correct"] for row in rows)
        if actual["correct"] != correct or not math.isclose(
            actual["overall_accuracy"], correct / len(rows), abs_tol=1e-12
        ):
            raise ValueError(f"{arm} overall score differs from labels")
        task_average = sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)
        if not math.isclose(actual["task_averaged_accuracy"], task_average, abs_tol=1e-12):
            raise ValueError(f"{arm} task average differs from labels")
        expected_types = {
            kind: {"correct": sum(values), "total": len(values)}
            for kind, values in sorted(grouped.items())
        }
        abstention = [row[f"{arm}_correct"] for row in rows if "_abs" in row["question_id"]]
        if actual["by_type"] != expected_types or actual["abstention"] != {
            "correct": sum(abstention),
            "total": len(abstention),
        }:
            raise ValueError(f"{arm} breakdown differs from labels")

    paired = {
        "memory_only_correct": sum(
            row["memory_correct"] and not row["baseline_correct"] for row in rows
        ),
        "baseline_only_correct": sum(
            row["baseline_correct"] and not row["memory_correct"] for row in rows
        ),
        "both_correct": sum(row["baseline_correct"] and row["memory_correct"] for row in rows),
        "both_wrong": sum(
            not row["baseline_correct"] and not row["memory_correct"] for row in rows
        ),
    }
    if summary["paired"] != paired or sum(paired.values()) != len(rows):
        raise ValueError("published paired counts differ from labels")


def main() -> None:
    root = Path(__file__).parent
    verify(
        root / "longmemeval_official_gpt4o_manifest.json",
        root / "longmemeval_official_gpt4o_labels.jsonl",
        root / "longmemeval_official_gpt4o_summary.json",
    )
    print("Published official GPT-4o evaluator results verified")


if __name__ == "__main__":
    main()
