"""Prepare and verify inputs for LongMemEval's unmodified GPT-4o evaluator.

The official evaluator is run on each small hypothesis chunk. Raw responses and
judgments stay under ignored ``data/``; only verified aggregate metrics may be
published. This does not submit a result to the benchmark leaderboard.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

MODEL = "gpt-4o-2024-08-06"
ARMS = ("baseline", "memory")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _unique_by_id(rows: list[dict], label: str) -> dict[str, dict]:
    by_id = {row["question_id"]: row for row in rows}
    if len(by_id) != len(rows) or not all(isinstance(key, str) and key for key in by_id):
        raise ValueError(f"{label} has missing or duplicate question IDs")
    return by_id


def prepare(
    dataset: Path,
    answers: Path,
    evaluator: Path,
    output: Path,
    *,
    chunk_size: int = 25,
    expected_count: int = 500,
) -> dict:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    references = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(references, list) or len(references) != expected_count:
        raise ValueError(f"dataset must contain {expected_count} questions")
    answer_rows = _jsonl(answers)
    ref_by_id = _unique_by_id(references, "dataset")
    answer_by_id = _unique_by_id(answer_rows, "answers")
    if set(ref_by_id) != set(answer_by_id):
        raise ValueError("raw answers do not cover exactly the dataset question IDs")
    for question_id, reference in ref_by_id.items():
        row = answer_by_id[question_id]
        if row["question_type"] != reference["question_type"]:
            raise ValueError(f"question type differs for {question_id}")
        if row["reference_answer"] != reference["answer"]:
            raise ValueError(f"reference answer differs for {question_id}")
        for field in ("baseline_answer", "memory_answer"):
            if not isinstance(row[field], str):
                raise ValueError(f"{field} is not text for {question_id}")

    manifest = {
        "dataset_sha256": _sha256(dataset),
        "raw_answers_sha256": _sha256(answers),
        "official_evaluator_sha256": _sha256(evaluator),
        "grader_model": MODEL,
        "questions": len(references),
        "chunk_size": chunk_size,
        "question_ids": [row["question_id"] for row in references],
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("existing grading manifest differs; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for offset in range(0, len(references), chunk_size):
            path = output / f"{arm}-{offset:04d}.jsonl"
            lines = [
                json.dumps(
                    {
                        "question_id": reference["question_id"],
                        "hypothesis": answer_by_id[reference["question_id"]][f"{arm}_answer"],
                    },
                    ensure_ascii=False,
                )
                for reference in references[offset : offset + chunk_size]
            ]
            content = "\n".join(lines) + "\n"
            if path.exists() and path.read_text(encoding="utf-8") != content:
                raise ValueError(f"existing hypotheses differ: {path}")
            path.write_text(content, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def summarize(dataset: Path, answers: Path, evaluator: Path, output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for path, key in (
        (dataset, "dataset_sha256"),
        (answers, "raw_answers_sha256"),
        (evaluator, "official_evaluator_sha256"),
    ):
        if _sha256(path) != manifest[key]:
            raise ValueError(f"input changed since preparation: {path}")
    references = json.loads(dataset.read_text(encoding="utf-8"))
    by_id = _unique_by_id(references, "dataset")
    labels: dict[str, dict[str, bool]] = {}
    for arm in ARMS:
        arm_labels: dict[str, bool] = {}
        for offset in range(0, len(references), manifest["chunk_size"]):
            path = output / f"{arm}-{offset:04d}.jsonl"
            inputs = _jsonl(path)
            graded = _jsonl(Path(str(path) + ".eval-results-gpt-4o"))
            if len(inputs) != len(graded):
                raise ValueError(f"incomplete official grading: {path}")
            for source, result in zip(inputs, graded, strict=True):
                if (
                    source["question_id"] != result["question_id"]
                    or source["hypothesis"] != result["hypothesis"]
                ):
                    raise ValueError(f"official output does not match hypothesis: {path}")
                question_id = result["question_id"]
                if question_id not in by_id or question_id in arm_labels:
                    raise ValueError(f"unexpected or repeated question ID: {question_id}")
                judgment = result["autoeval_label"]
                if judgment["model"] != MODEL or type(judgment["label"]) is not bool:
                    raise ValueError(f"invalid grader label for {question_id}")
                arm_labels[question_id] = judgment["label"]
        if set(arm_labels) != set(by_id):
            raise ValueError(f"{arm} grading does not cover all questions")
        labels[arm] = arm_labels

    summary = {"grader_model": MODEL, "questions": len(references), "arms": {}, "paired": {}}
    for arm in ARMS:
        by_type: dict[str, list[bool]] = defaultdict(list)
        for reference in references:
            by_type[reference["question_type"]].append(labels[arm][reference["question_id"]])
        abstention = [
            labels[arm][row["question_id"]] for row in references if "_abs" in row["question_id"]
        ]
        correct = sum(labels[arm].values())
        summary["arms"][arm] = {
            "correct": correct,
            "overall_accuracy": correct / len(references),
            "task_averaged_accuracy": sum(sum(v) / len(v) for v in by_type.values()) / len(by_type),
            "abstention": {"correct": sum(abstention), "total": len(abstention)},
            "by_type": {
                kind: {"correct": sum(values), "total": len(values)}
                for kind, values in sorted(by_type.items())
            },
        }
    summary["paired"] = {
        "memory_only_correct": sum(
            labels["memory"][qid] and not labels["baseline"][qid] for qid in by_id
        ),
        "baseline_only_correct": sum(
            labels["baseline"][qid] and not labels["memory"][qid] for qid in by_id
        ),
        "both_correct": sum(labels["baseline"][qid] and labels["memory"][qid] for qid in by_id),
        "both_wrong": sum(
            not labels["baseline"][qid] and not labels["memory"][qid] for qid in by_id
        ),
    }
    public_labels = [
        {
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "baseline_correct": labels["baseline"][row["question_id"]],
            "memory_correct": labels["memory"][row["question_id"]],
        }
        for row in references
    ]
    (output / "labels.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in public_labels), encoding="utf-8"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "summarize"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(
            args.dataset, args.answers, args.evaluator, args.output, chunk_size=args.chunk_size
        )
    else:
        result = summarize(args.dataset, args.answers, args.evaluator, args.output)
    print(
        json.dumps(
            result if args.action == "summarize" else {"questions": result["questions"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
