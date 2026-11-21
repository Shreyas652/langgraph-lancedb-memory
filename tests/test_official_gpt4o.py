"""The official-grader adapter checks IDs, inputs, and published aggregates."""

import json

import pytest

from evals.official_gpt4o import MODEL, prepare, summarize
from evals.verify_official_gpt4o import verify


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_official_grader_inputs_and_summary(tmp_path):
    references = [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "answer": "A",
        },
        {
            "question_id": "q2_abs",
            "question_type": "single-session-user",
            "answer": "unanswerable",
        },
    ]
    answers = [
        {
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "reference_answer": row["answer"],
            "baseline_answer": f"baseline-{i}",
            "memory_answer": f"memory-{i}",
        }
        for i, row in enumerate(references)
    ]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(references), encoding="utf-8")
    raw = tmp_path / "answers.jsonl"
    _write_jsonl(raw, answers)
    evaluator = tmp_path / "evaluate_qa.py"
    evaluator.write_text("# pinned official evaluator\n", encoding="utf-8")
    output = tmp_path / "graded"
    manifest = prepare(dataset, raw, evaluator, output, chunk_size=1, expected_count=2)
    assert manifest["questions"] == 2
    assert len(list(output.glob("baseline-*.jsonl"))) == 2

    labels = {"baseline": [False, True], "memory": [True, True]}
    for arm in labels:
        for offset, correct in enumerate(labels[arm]):
            path = output / f"{arm}-{offset:04d}.jsonl"
            source = json.loads(path.read_text(encoding="utf-8"))
            source["autoeval_label"] = {"model": MODEL, "label": correct}
            _write_jsonl(output / f"{path.name}.eval-results-gpt-4o", [source])

    summary = summarize(dataset, raw, evaluator, output)
    assert summary["arms"]["baseline"]["correct"] == 1
    assert summary["arms"]["memory"]["correct"] == 2
    assert summary["paired"] == {
        "memory_only_correct": 1,
        "baseline_only_correct": 0,
        "both_correct": 1,
        "both_wrong": 0,
    }
    assert summary["arms"]["memory"]["abstention"] == {"correct": 1, "total": 1}
    public_labels = [
        json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()
    ]
    assert public_labels == [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "baseline_correct": False,
            "memory_correct": True,
        },
        {
            "question_id": "q2_abs",
            "question_type": "single-session-user",
            "baseline_correct": True,
            "memory_correct": True,
        },
    ]
    verify(output / "manifest.json", output / "labels.jsonl", output / "summary.json")
    summary["arms"]["memory"]["correct"] = 1
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="overall score differs"):
        verify(output / "manifest.json", output / "labels.jsonl", output / "summary.json")


def test_official_grader_rejects_different_reference_and_tampered_label(tmp_path):
    references = [{"question_id": "q", "question_type": "multi-session", "answer": "A"}]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(references), encoding="utf-8")
    raw = tmp_path / "answers.jsonl"
    answers = [
        {
            "question_id": "q",
            "question_type": "multi-session",
            "reference_answer": "B",
            "baseline_answer": "x",
            "memory_answer": "y",
        }
    ]
    _write_jsonl(raw, answers)
    evaluator = tmp_path / "evaluate_qa.py"
    evaluator.write_text("# pinned official evaluator\n", encoding="utf-8")
    output = tmp_path / "graded"
    with pytest.raises(ValueError, match="reference answer differs"):
        prepare(dataset, raw, evaluator, output, expected_count=1)
    answers[0]["reference_answer"] = "A"
    _write_jsonl(raw, answers)
    prepare(dataset, raw, evaluator, output, expected_count=1)
    for arm in ("baseline", "memory"):
        result = json.loads((output / f"{arm}-0000.jsonl").read_text(encoding="utf-8"))
        result["autoeval_label"] = {"model": MODEL, "label": True}
        _write_jsonl(output / f"{arm}-0000.jsonl.eval-results-gpt-4o", [result])
    evaluator.write_text("# changed evaluator\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        summarize(dataset, raw, evaluator, output)
