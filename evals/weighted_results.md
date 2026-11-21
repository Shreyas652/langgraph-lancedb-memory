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

| Retrieval | Vector weight | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vector | n/a | 60.0% | 84.0% | 92.0% | 0.716 |
| Hybrid | 0.5 | 82.0% | 96.0% | 100.0% | 0.876 |
| Hybrid | 0.7 | 70.0% | 94.0% | 96.0% | 0.802 |
| Hybrid | 0.9 | 60.0% | 86.0% | 96.0% | 0.726 |

The public-data pilot reverses the preference seen on the synthetic questions: equal-weight hybrid retrieved the most evidence sessions. The benchmark session text is sent to BGE as one document, so the model's context limit can truncate long sessions; this is a retrieval baseline, not a full conversational-memory system. The full non-abstention split and answer-generation evaluation remain to be run before making a headline benchmark claim.

Reproduce the pilot after downloading `longmemeval_s_cleaned.json` to `data/` and caching the BGE model:

```sh
uv run --offline python evals/longmemeval_retrieval.py --dataset data/longmemeval_s_cleaned.json --limit 50 --offline
```
