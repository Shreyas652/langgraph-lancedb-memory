# Retrieval evaluation

Synthetic, hand-written data: 54 memories, three users, four sessions per user, 45 questions. The measured embedder is FastEmbed BAAI/bge-small-en-v1.5. Each search is filtered to the question's user before ranking. Recall@k means at least one gold memory occurs in the first k results. MRR uses the first gold rank. Stale rate is measured only on updated questions and means a superseded memory appeared ahead of the current answer in the top five. Leak rate means any returned memory belonged to a different user. Latency includes embedding and local search; it varies by machine.

| Mode | Type | N | Recall@1 | Recall@3 | Recall@5 | MRR | Stale rate | Leak rate | Median ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| recency | all | 45 | 6.7% | 20.0% | 33.3% | 0.152 | 0.0% | 0.0% | 12.7 |
| recency | direct | 15 | 0.0% | 0.0% | 20.0% | 0.040 | n/a | 0.0% | 13.5 |
| recency | paraphrase | 12 | 0.0% | 0.0% | 0.0% | 0.000 | n/a | 0.0% | 13.8 |
| recency | updated | 9 | 0.0% | 66.7% | 100.0% | 0.361 | 0.0% | 0.0% | 12.4 |
| recency | cross_user | 9 | 33.3% | 33.3% | 33.3% | 0.333 | n/a | 0.0% | 11.7 |
| vector | all | 45 | 97.8% | 100.0% | 100.0% | 0.989 | 0.0% | 0.0% | 23.0 |
| vector | direct | 15 | 93.3% | 100.0% | 100.0% | 0.967 | n/a | 0.0% | 23.9 |
| vector | paraphrase | 12 | 100.0% | 100.0% | 100.0% | 1.000 | n/a | 0.0% | 23.5 |
| vector | updated | 9 | 100.0% | 100.0% | 100.0% | 1.000 | 0.0% | 0.0% | 22.4 |
| vector | cross_user | 9 | 100.0% | 100.0% | 100.0% | 1.000 | n/a | 0.0% | 22.2 |
| hybrid | all | 45 | 77.8% | 88.9% | 97.8% | 0.839 | 0.0% | 0.0% | 29.4 |
| hybrid | direct | 15 | 100.0% | 100.0% | 100.0% | 1.000 | n/a | 0.0% | 29.0 |
| hybrid | paraphrase | 12 | 33.3% | 66.7% | 91.7% | 0.503 | n/a | 0.0% | 30.8 |
| hybrid | updated | 9 | 77.8% | 88.9% | 100.0% | 0.856 | 0.0% | 0.0% | 29.0 |
| hybrid | cross_user | 9 | 100.0% | 100.0% | 100.0% | 1.000 | n/a | 0.0% | 28.4 |
| hybrid+history | all | 45 | 68.9% | 84.4% | 91.1% | 0.773 | 55.6% | 0.0% | 27.9 |
| hybrid+history | direct | 15 | 100.0% | 100.0% | 100.0% | 1.000 | n/a | 0.0% | 27.8 |
| hybrid+history | paraphrase | 12 | 33.3% | 50.0% | 75.0% | 0.439 | n/a | 0.0% | 27.7 |
| hybrid+history | updated | 9 | 33.3% | 88.9% | 88.9% | 0.611 | 55.6% | 0.0% | 32.4 |
| hybrid+history | cross_user | 9 | 100.0% | 100.0% | 100.0% | 1.000 | n/a | 0.0% | 28.0 |
