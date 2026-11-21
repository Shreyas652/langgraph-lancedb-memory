# Memory quality and scale evaluation

## What is measured

The [weighted retrieval report](weighted_results.md) contains the full
470-question LongMemEval-S cleaned **evidence retrieval** run. It uses one
LanceDB row per conversation session and reports any-gold-session Hit@k. The
0.5 vector weight led that public-data run (85.7% Hit@1 versus 77.7% for
vector-only), while the same equal-weight fusion hurt paraphrases in the small
synthetic set (33.3% versus 100% vector-only Hit@1). A weight is therefore a
dataset-dependent setting, not a universal improvement.

The paired answer runner, `python -m evals.longmemeval_answers`, evaluates
LongMemEval-S questions twice with the same Bedrock reader model: first with
the question only, then with up to `k` retrieved history sessions. A separate
Bedrock judge sees the benchmark reference only after both answers are
generated. It includes abstention questions and logs raw answers, verdicts,
retrieved session IDs, and token usage to an ignored local checkpoint. The
runner reports paired accuracy, wins and losses, question-type accuracy, and
strict all-evidence inclusion in the actual answer context. The judge prompt
follows the public benchmark's task categories but uses Bedrock Claude rather
than its published GPT-4o judge, so these results must be labeled
**Bedrock-judged**, not an official LongMemEval leaderboard score. Automatic
judgments can be wrong; inspect a sample of disagreements before using this
to make product decisions. The context budget is explicit and complete
sessions that do not fit are skipped, not silently truncated.

## Published GPT-4o evaluator on the same 500 answers

The saved answers from the run below were graded again with LongMemEval's
**unmodified** [`evaluate_qa.py`](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py)
at commit `9e0b455f4ef0e2ab8f2e582289761153549043fc`. Its `gpt-4o` option
selects `gpt-4o-2024-08-06`, temperature zero, and the benchmark's
task-specific grading prompts. The 500 baseline and 500 memory hypotheses
were split into 25-question chunks for resumability; no answer was regenerated
or changed. The official [`print_qa_metrics.py`](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/print_qa_metrics.py)
independently reproduced both overall and task-averaged scores from the
combined evaluator outputs.
The evaluator ran in an isolated Python 3.10.11 environment with the
benchmark's `requirements-lite.txt` (`openai==1.35.1`, `numpy==1.26.3`)
and `httpx==0.27.2` for compatibility with that OpenAI client. The local
[PowerShell runner](run_official_gpt4o.ps1) loads an ignored, user-encrypted
API key and runs the unchanged evaluator on every prepared chunk.

| Measure | Without memory | With retrieved memory |
| --- | ---: | ---: |
| Overall accuracy | 31 / 500 (6.2%) | 351 / 500 (70.2%) |
| Task-averaged accuracy | 5.2% | 71.7% |
| Abstention accuracy | 30 / 30 (100%) | 30 / 30 (100%) |

The memory arm alone was correct on 320 questions, the baseline alone on zero;
both were correct on 31 and both were wrong on 149. The 64.0 percentage-point
overall gap measures these saved Haiku 4.5 answers under the published GPT-4o
evaluator. It is **not a leaderboard submission**, a production accuracy
estimate, or evidence that the agent's fact extractor works: this experiment
stored whole history sessions as memory rows.

| LongMemEval question type | N | Without memory | With memory |
| --- | ---: | ---: | ---: |
| Knowledge update | 78 | 6 (7.7%) | 61 (78.2%) |
| Multi-session | 133 | 12 (9.0%) | 82 (61.7%) |
| Single-session assistant | 56 | 0 (0%) | 56 (100%) |
| Single-session preference | 30 | 0 (0%) | 13 (43.3%) |
| Single-session user | 70 | 7 (10.0%) | 63 (90.0%) |
| Temporal reasoning | 133 | 6 (4.5%) | 76 (57.1%) |

The [summary](longmemeval_official_gpt4o_summary.json),
[manifest with source and input hashes](longmemeval_official_gpt4o_manifest.json),
and [text-free per-question labels](longmemeval_official_gpt4o_labels.jsonl)
are public. CI recomputes their counts and scores independently. Raw answers,
prompts, and evaluator outputs remain in ignored local `data/`. The evaluator
file SHA-256 is `5085eb9ae08b91c6c4f97c6bea9b8c2d6e55f9ec72c27734cafa0a04f00a430a`;
the saved answers SHA-256 is `8e104deaadf494d56da96194d5e295f48879b34dd3b22fb9e5db4ab3c6071d70`.
The [preparation and verification code](official_gpt4o.py) checks all 500 IDs,
question types, reference answers, grader labels, and hashes before publishing
aggregates. Automatic GPT-4o judgments can still be mistaken on individual
questions.

## Separate Claude-judged 500-question result

The run completed all **500 LongMemEval-S cleaned questions**, including 30
abstentions, on 21 November 2026. It indexed 23,867 full history sessions in
25-question chunks. The dataset SHA-256 was
`d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
Both arms used Bedrock `us.anthropic.claude-haiku-4-5-20251001-v1:0` at
temperature zero. The separate judge was `us.anthropic.claude-sonnet-4-6`.
The memory arm retrieved up to five sessions with the 0.5-weight hybrid
retriever and admitted complete sessions up to an 80,000-character budget.
The baseline saw the same question and current date without history.

| Measure | Without memory | With retrieved memory |
| --- | ---: | ---: |
| Bedrock-judged correct answers | 26 / 500 (5.2%) | 333 / 500 (66.6%) |

Memory alone won 309 paired questions; baseline alone won two. Both were
correct on 24 and both were wrong on 165. The 61.4 percentage-point gap is
specific to this dataset, prompt, retriever, context budget, reader, and
judge. It is not a production user accuracy estimate.

| LongMemEval question type | N | Without memory | With memory |
| --- | ---: | ---: | ---: |
| Knowledge update | 78 | 6 (7.7%) | 56 (71.8%) |
| Multi-session | 133 | 10 (7.5%) | 76 (57.1%) |
| Single-session assistant | 56 | 0 (0%) | 56 (100%) |
| Single-session preference | 30 | 0 (0%) | 14 (46.7%) |
| Single-session user | 70 | 6 (8.6%) | 63 (90.0%) |
| Temporal reasoning | 133 | 4 (3.0%) | 68 (51.1%) |

For the 470 answerable questions, all gold evidence sessions appeared in the
top five on 414 (88.1%); at least one gold session appeared on 457 (97.2%).
After applying the context budget, all gold sessions remained on 408
(86.8%). These evidence rates explain some answer failures but do not prove
that the model used the retrieved evidence correctly.

The committed [summary](longmemeval_answer_summary.json),
[manifest](longmemeval_answer_manifest.json), and
[text-free per-question verdicts](longmemeval_answer_labels.jsonl) are
independently checked by `python -m evals.verify_answer_evidence` in CI.
With the public dataset and exact source archive available, the verifier also
checks their hashes. The raw answers, judge text, and instance log are stored
only in the local ignored archive
`data/aws-answer/longmemeval-answer-results-20261121.zip` (SHA-256
`38755868a5908e24265d0d8685ba439e781cca8b2712182f49803799035e967d`).
The archived source was built from commit `d7ccb02`; its SHA-256 was
`c6c492d6a4d75f876b1b4f313d8486c06b7aa2ccd50a562362a56a9a3e85c46a`.
Windows archive line endings differ from the Git blob, and the normalized
runner content was checked to be identical. The AWS instance exited zero
after writing a `COMPLETE` summary; all temporary AWS resources were removed
after the downloaded checkpoint passed an independent 500-case audit.

This evaluation indexes full sessions as memory rows. It does **not** test
the agent's fact extractor on those histories, natural-language temporal
query parsing, concurrent writers, or real user satisfaction. The judge is
not the benchmark's [published GPT-4o evaluator](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py),
and automatic judgments may contain errors. A fixed sample of wins, losses,
and misses was inspected, but the 500 judgments have not all been reviewed by
humans. This is a reproducible separate Bedrock-judged result. The GPT-4o
evaluation above uses the same saved answers and the benchmark's published
evaluator.

## Fact updates and time

`MemoryStore.add` uses `(user_id, topic)` as its explicit fact identity. An
exact duplicate active fact returns the existing ID; a changed value
supersedes the prior active row in one LanceDB merge. A same-timestamp
conflict is rejected, including historical backfills. `get_topic_at` returns
the latest fact at an explicit timezone-aware timestamp and `compare_topic`
returns both states for a before/after question. Historical dates are
observation times supplied by the caller; this API does not infer natural
language dates or semantic equivalence between differently worded facts.
It is a local single-process store and still needs concurrency control for
multi-writer deployment.

The read-only CLI exposes the same explicit comparison without invoking a
model:

```sh
uv run python -m agent_memory.cli compare --user alice --topic home_city --before 2026-01-15T00:00:00+00:00 --after 2026-02-15T00:00:00+00:00
```

## 100,000-row index comparison

`python -m evals.scale_100k` generated 100,000 deterministic synthetic
384-dimensional normalized vectors with the same schema as the memory store.
The 100 query vectors were sampled from stored rows. The script compared
exact scans against an `IVF_FLAT` index with 256 partitions on the same data.
`Recall@10` is index overlap with exact top 10, not semantic answer quality.
The exact median shown is the post-index exact scan in the same run; p50
times are single-host, warm-cache observations, not service-level guarantees.

### Local Windows run

| Probes | Exact p50 ms | Indexed p50 ms | Indexed p95 ms | Recall@10 | Top-1 agreement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 120.2 | 25.8 | 41.0 | 28.5% | 100% |
| 128 | 76.5 | 24.4 | 37.3 | 76.8% | 100% |
| 256 | 107.3 | 37.9 | 51.8 | 100% | 100% |

Probing all partitions matched exact top 10 on these 100 queries and had a
2.8x lower median than the within-run exact scan. The IVF_FLAT index took
4.0 to 6.6 seconds to build. The raw local summaries are
[`16 probes`](scale_100k_summary.json),
[`128 probes`](scale_100k_nprobes128.json), and
[`256 probes`](scale_100k_nprobes256.json).

### AWS c7i.4xlarge run

The same committed script, seed, row count, dimensions, 100 sampled queries,
and probe settings ran on Amazon Linux 2023 with Python 3.11. This is a
separate measurement, not a repeat on identical hardware.

| Probes | Exact p50 ms | Indexed p50 ms | Indexed p95 ms | Recall@10 | Top-1 agreement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 19.8 | 12.0 | 14.4 | 25.5% | 100% |
| 128 | 17.4 | 18.8 | 21.2 | 78.7% | 100% |
| 256 | 17.1 | 25.2 | 27.9 | 100% | 100% |

The 256-probe index was **slower than exact search on AWS** despite matching
its top 10 on all 100 queries. The 16-probe index was faster but missed most
of the exact top-10 neighbors. Index build time was 1.6 to 1.7 seconds and
the table plus index occupied about 315 MB. This reversal from the local
Windows timing means these runs do not justify making ANN the default for
100k memories. Benchmark on the target hardware with a required recall
threshold before choosing index settings. The raw AWS summaries are
[`16 probes`](scale_100k_aws_nprobes16.json),
[`128 probes`](scale_100k_aws_nprobes128.json), and
[`256 probes`](scale_100k_aws_nprobes256.json).
The AWS run used source archive SHA-256
`ac4a6d9cd4e8419a3c60cdf84271de4228c972caf622d71398115d52ad6baffc`
from commit `f1b12b8`. Its three summaries and successful instance log are
preserved in the ignored local archive
`data/aws-scale/aws-scale-results-20261121.zip` (SHA-256
`560036900be89708c9de528cbe9f557cf9d148719312b1cf5de12b7efb0df494`).
The temporary instance, role, profile, security group, and bucket were
removed after the archive passed verification.

These seeded query vectors guarantee an exact stored match, which makes
top-1 easy and limits generalization to real embedding queries. The table is
synthetic and has no concurrent writes, network latency, or multi-user
filters. Each setting was run once on each host, so the timing comparison
also lacks repeated-run variance estimates.

Reproduce with a clean environment:

```sh
uv sync --locked --extra bedrock
uv run --offline python -m evals.scale_100k --rows 100000 --queries 100 --partitions 256 --nprobes 256 --output evals/scale_100k_nprobes256.json
uv run python -m evals.longmemeval_answers --dataset data/longmemeval_s_cleaned.json --output data/answer-full --profile YOUR_AWS_PROFILE --region us-east-1 --reader-model us.anthropic.claude-haiku-4-5-20251001-v1:0 --judge-model us.anthropic.claude-sonnet-4-6
uv run --offline python -m evals.verify_answer_evidence
```

The public dataset and BGE model must be downloaded separately. The answer
run requires authenticated Bedrock access and incurs model charges. Neither
the dataset nor raw answers are committed.
