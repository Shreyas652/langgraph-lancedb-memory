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
All results below are actual local Windows measurements. `Recall@10` is
index overlap with exact top 10, not semantic answer quality. The exact
median shown is the post-index exact scan in the same run; p50 times are
single-host, warm-cache observations, not service-level guarantees.

| Probes | Exact p50 ms | Indexed p50 ms | Indexed p95 ms | Recall@10 | Top-1 agreement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 120.2 | 25.8 | 41.0 | 28.5% | 100% |
| 128 | 76.5 | 24.4 | 37.3 | 76.8% | 100% |
| 256 | 107.3 | 37.9 | 51.8 | 100% | 100% |

The 16-probe setting is unsuitable when top-10 completeness matters. Probing
all partitions matched exact top 10 on these 100 queries and had a 2.8x
lower median than the within-run exact scan. The IVF_FLAT index took 4.0 to
6.6 seconds to build; the stored table and index occupied about 315 MB.
These seeded query vectors guarantee an exact stored match, which makes
top-1 easy and limits generalization to real embedding queries. The table is
synthetic and has no concurrent writes, network latency, or multi-user
filters. The raw summaries are
[`16 probes`](scale_100k_summary.json),
[`128 probes`](scale_100k_nprobes128.json), and
[`256 probes`](scale_100k_nprobes256.json).

Reproduce with a clean environment:

```sh
uv sync --locked --extra bedrock
uv run --offline python -m evals.scale_100k --rows 100000 --queries 100 --partitions 256 --nprobes 256 --output evals/scale_100k_nprobes256.json
uv run --offline python -m evals.longmemeval_answers --dataset data/longmemeval_s_cleaned.json --output data/answer-full --profile YOUR_AWS_PROFILE --region us-east-1 --reader-model YOUR_BEDROCK_READER_ID --judge-model YOUR_BEDROCK_JUDGE_ID
```

The public dataset and BGE model must be downloaded separately. The answer
run requires authenticated Bedrock access and incurs model charges. Neither
the dataset nor raw answers are committed.
