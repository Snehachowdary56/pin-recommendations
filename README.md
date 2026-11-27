# Pin Recommendations: graph-based retrieval and a neural ranker in PyTorch

A two-stage content recommendation system for a discovery feed, built in **PyTorch** with experiments tracked in
**MLflow**: graph-based embeddings (**LightGCN**) and matrix factorization trained with a ranking loss (BPR),
a content model from text embeddings for new pins, and a neural **ranker**, evaluated offline with
recall@20, NDCG@20, cold-start recall and catalog coverage.

> **Data note:** all data is **generated sample data** (3,000 users, 6,000 pins in 24 topics, about 88,000 saves).
> It is not Pinterest data. Known structure is built in (users have 2-3 interests, popularity has a long tail,
> 8% of pins are brand new) so results can be checked.

## How it works

```
                      ┌─ LightGCN on the user–pin save graph ── top 200 pins ─┐
user ─► Stage 1       │                                                       ├─► ~250 candidates ─► Stage 2: neural ranker ─► top 20
        (retrieval)   └─ "fresh pins": text embeddings over pins with no saves ─ top 50 ─┘        (graph, text, topic, popularity features)
```

1. **Problem definition.** Recommend 20 pins a user will save next. Split each user's saves by time: first 70% to
   train, next 10% to validate, last 20% to test, so the model never sees the future.
2. **Graph-based retrieval.** LightGCN learns user and pin embeddings by passing them along the save graph
   (2 layers, symmetric-normalized adjacency), trained with BPR loss: a saved pin should score above a random one.
3. **Cold start.** A graph model can never recommend a pin with no saves. A "fresh pins" source ranks unsaved pins
   by text embeddings (TF-IDF + SVD of the pin title) against the user's content profile.
4. **Ranking.** A small PyTorch MLP scores each candidate from 7 features (graph score and rank, text score and
   rank, log popularity, is-new flag, the user's affinity for the pin's topic), trained on the validation period.
   8% of older pins are hidden from the graph in training so the ranker learns how pins without saves behave.
5. **Fresh slots.** Optionally reserve 3 of the 20 slots for the best-scored fresh pins.

## Results ([`results/RESULTS.md`](results/RESULTS.md))

| Model | Recall@20 | NDCG@20 | Cold-start recall@20 | Catalog coverage |
|---|---|---|---|---|
| Popularity baseline | 0.036 | 0.025 | 0.000 | 0.4% |
| Matrix factorization (BPR) | 0.140 | 0.111 | 0.000 | 28.2% |
| LightGCN, 2 layers (graph-based) | 0.176 | 0.142 | 0.000 | 14.5% |
| Content: pin title embeddings | 0.043 | 0.027 | 0.036 | 88.6% |
| Two-stage: graph + fresh-pin candidates → neural ranker | **0.189** | **0.153** | 0.000 | 15.6% |
| Two-stage ranker + 3 of 20 slots for fresh pins | 0.185 | 0.151 | **≈0.04** | **≈29%** |

What I learned:

- **Graph propagation helps.** LightGCN beats matrix factorization (recall@20 0.176 vs 0.140) with the same embedding size.
- **The ranker adds on top of retrieval** (0.189 vs 0.176), mainly from topic affinity and popularity features.
- **Accuracy and freshness pull against each other.** The ranker alone never puts a new pin in the top 20.
  Reserving 3 slots for fresh pins makes new pins discoverable and doubles catalog coverage for a small recall cost.
  Which matters more is a product decision for an online A/B test, not an offline metric.
- **Limit:** pin titles here only reveal the topic. With real text or image embeddings (for example a vision model
  on pin images) the content model, and cold-start recall, should do much better.

## MLflow

Every run logs its parameters (model, embedding size, layers, epochs, learning rate, candidate counts, features)
and metrics (recall, NDCG, cold-start recall, coverage, training time) to `mlflow.db`.

```
pip install mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 and compare the 7 runs side by side.

## Run it (Windows Command Prompt)

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
python -m pinrec
```

`pytest -q` shows **10 passed**. `python -m pinrec` trains every model on CPU in about 30 seconds and writes
`results/RESULTS.md`, `results/metrics.csv` and `mlflow.db`. Small run-to-run differences (third decimal) come
from multi-threaded CPU math.

## Tests

- Split is temporal; new pins appear only in the test period.
- The graph adjacency is symmetric and normalized as D^-1/2 A D^-1/2.
- LightGCN layers mix in neighbour embeddings; BPR loss falls during training.
- top-k never repeats a pin the user already saved; recall, NDCG, cold-start recall and coverage match hand-computed values.
- Fresh candidates only add unsaved pins; fresh slots work as described.
- End-to-end run on a small dataset beats popularity and logs to MLflow.

## Files

| File | What it does |
|---|---|
| `pinrec/data.py` | Generates users, pins (topic, title, new flag) and saves; temporal split |
| `pinrec/models.py` | Matrix factorization, LightGCN, BPR training, text embeddings |
| `pinrec/rank.py` | Candidate generation, ranking features, PyTorch MLP ranker, fresh slots |
| `pinrec/evaluate.py` | top-k, recall@k, NDCG@k, cold-start recall, catalog coverage |
| `pinrec/__main__.py` | Runs every experiment, logs to MLflow, writes results |
