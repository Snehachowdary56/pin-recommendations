# Results (generated sample data)

3,000 users, 6,000 pins in 24 topics, 88,343 saves.
Temporal split per user: first 70% train, next 10% validation (trains the ranker), last 20% test.
Another 8% of older pins are hidden from the graph during training so the ranker can learn to rank pins with no saves.
8% of pins are new (no saves before the test period); they are 18% of test saves.

| Model | Recall@20 | NDCG@20 | Cold-start recall@20 | Catalog coverage |
|---|---|---|---|---|
| Popularity baseline | 0.036 | 0.025 | 0.000 | 0.4% |
| Matrix factorization (BPR) | 0.140 | 0.111 | 0.000 | 28.2% |
| LightGCN, 2 layers (graph-based) | 0.176 | 0.142 | 0.000 | 14.5% |
| Content: pin title embeddings | 0.043 | 0.027 | 0.036 | 88.6% |
| Two-stage, graph candidates only (ablation) | 0.189 | 0.153 | 0.000 | 13.9% |
| Two-stage: graph + fresh-pin candidates → neural ranker | 0.189 | 0.153 | 0.000 | 15.6% |
| **Two-stage ranker + 3 of 20 slots for fresh pins** | 0.185 | 0.151 | 0.044 | 29.5% |

- Graph propagation helps: LightGCN beats plain matrix factorization (recall@20 0.176 vs 0.140).
- The two-stage system (LightGCN + fresh-pin candidates, then a neural ranker with graph, text, topic and popularity
  features) beats LightGCN alone: recall@20 0.189 vs 0.176, NDCG@20 0.153 vs 0.142.
- Graph-only models never recommend a new pin (cold-start recall 0), and even the ranker keeps new pins out of the
  top 20 because established pins score higher. Reserving 3 of 20 slots for the best fresh pins raises cold-start
  recall to 0.044 and catalog coverage from 15.6% to 29.5%, for a small cost in
  overall recall (0.189 → 0.185). That is a product decision (fresh content vs short-term accuracy),
  best settled with an online A/B test.
- Limit: pin titles here only reveal the topic, so the content model cannot tell which new pin in a topic a user
  will like; with real text or image embeddings, cold-start recall should be higher.
- Stage 1 keeps 63.2% of test saves among about 250 candidates per user; stage 2 orders them.
- Ranker training data: 0.75% of validation candidates were positives (class weighting used).
- BPR loss, last epoch: MF 0.065, LightGCN 0.067.

Every run is tracked in MLflow (`mlflow.db`; open with `mlflow ui --backend-store-uri sqlite:///mlflow.db`): parameters, metrics and training time.
