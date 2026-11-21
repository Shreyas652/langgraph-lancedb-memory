# Weighted retrieval: exploratory results

## Local synthetic data

The existing 45-question hand-written evaluation was rerun with the cached BGE model. Only the hybrid vector weight changed; the same data, search depth, and lexical index were used. The figures below are Recall@1, so they do not imply answer accuracy.

| Retrieval | Vector weight | All 45 | Direct 15 | Paraphrase 12 | Updated 9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | n/a | 97.8% | 93.3% | 100.0% | 100.0% |
| Hybrid | 0.5 | 77.8% | 100.0% | 33.3% | 77.8% |
| Hybrid | 0.7 | 86.7% | 100.0% | 58.3% | 88.9% |
| Hybrid | 0.9 | 97.8% | 93.3% | 100.0% | 100.0% |

Equal-weight RRF made lexical rank as influential as vector rank. On reworded questions, lexical matches could outrank semantic matches. More vector weight mitigated that loss, but at 0.9 the exact-word advantage disappeared. These weights were inspected on the same small dataset, so selecting one as a default from these numbers would overfit.

## LongMemEval retrieval pilot

The official [LongMemEval cleaned small split](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) was downloaded locally (SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`). This pilot used the **first 50 non-abstention questions**, indexing 2,436 complete conversation sessions as separate LanceDB rows. Each question was isolated to its own history, and the benchmark's `answer_session_ids` supplied the gold evidence labels. The official retrieval protocol excludes abstention questions. This is a subset pilot, **not a full LongMemEval score or answer-accuracy result**.

| Retrieval | Vector weight | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | n/a | 60.0% | 84.0% | 92.0% | 0.716 |
| Hybrid | 0.5 | 82.0% | 96.0% | 100.0% | 0.876 |
| Hybrid | 0.7 | 70.0% | 94.0% | 96.0% | 0.802 |
| Hybrid | 0.9 | 60.0% | 86.0% | 96.0% | 0.726 |

The public-data pilot reverses the preference seen on the synthetic questions: equal-weight hybrid retrieved the most evidence sessions. The benchmark session text is sent to BGE as one document, so the model's context limit can truncate long sessions. The full retrieval run below tests whether that pilot finding holds across the entire eligible split.

Reproduce the pilot after downloading `longmemeval_s_cleaned.json` to `data/` and caching the BGE model:

```sh
uv run --offline python evals/longmemeval_retrieval.py --dataset data/longmemeval_s_cleaned.json --limit 50 --offline
```

## Full LongMemEval-S cleaned evidence retrieval

The full run evaluated all **470 non-abstention questions** in the official 500-question cleaned small split. It indexed **22,419 complete conversation sessions**, one row per session, and restricted every query to its own question's history. The dataset SHA-256 is the same as above. Four modes ran on each question: vector-only and hybrid RRF with vector weights 0.5, 0.7, and 0.9. The benchmark used FastEmbed BGE-small-en-v1.5, LanceDB, Python 3.11, and 25-question process chunks on an AWS c7i.4xlarge. Total elapsed time was 24 minutes 39 seconds, including indexing.

Here **Hit@k means at least one gold `answer_session_id` appeared in the first k results**. MRR@10 uses the rank of the first gold session, or zero for a miss. These are evidence-retrieval metrics, not the official `recall_all@k` metric, which requires every gold session, and not answer accuracy. Median latency includes query embedding and search, excludes indexing, and is hardware-dependent.

| Retrieval | Vector weight | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Median query ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vector | n/a | 77.7% | 93.4% | 97.0% | 0.846 | 11.2 |
| Hybrid | 0.5 | **85.7%** | **97.2%** | **98.5%** | **0.909** | 15.3 |
| Hybrid | 0.7 | 82.6% | 96.8% | 98.1% | 0.887 | 14.8 |
| Hybrid | 0.9 | 79.1% | 94.9% | 97.7% | 0.859 | 14.9 |

Equal-weight hybrid improves Hit@1 by **8.1 percentage points** over vector-only on this run (403 versus 365 questions). Increasing the vector weight to 0.7 or 0.9 reduces the public-data score despite helping the hand-written paraphrase subset. The 0.5 default therefore remains unchanged. This result applies to this session-level retriever and dataset; it does not establish end-to-end memory quality or justify a general weight recommendation.

The [summary](longmemeval_full_summary.json), [run manifest](longmemeval_full_manifest.json), and [per-question first-gold ranks](longmemeval_full_rankings.jsonl) are committed for audit. The raw dataset and model are not committed. The full local result archive contains all 19 checkpoint files and the instance log; an independent verifier checked unique question coverage, source and dataset hashes, session counts, ranks, and recomputed scores before the temporary AWS resources were removed.

The exact run source is preserved at commit `87c1bfa`. A later naming-only change labels the metric `Hit@k` in the CLI and code, matching the any-evidence calculation already used in that run.

To reproduce after downloading `longmemeval_s_cleaned.json` to `data/` and caching the BGE model in `data/models`:

```sh
uv run python -m evals.run_longmemeval_full --dataset data/longmemeval_s_cleaned.json --output data/longmemeval-full --chunk-size 25
```
