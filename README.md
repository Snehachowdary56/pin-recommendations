# Pin Recommendations: graph-based retrieval, a neural ranker, ads CTR prediction and visual search

A two-stage content recommendation system for a discovery feed, built in **PyTorch** with experiments tracked in
**MLflow**: graph-based embeddings (**LightGCN**) and matrix factorization trained with a ranking loss (BPR),
a content model from text embeddings for new pins, and a neural **ranker**, evaluated offline with
recall@20, NDCG@20, cold-start recall and catalog coverage. It also covers **ads ranking** (CTR prediction,
calibration, position bias and an auction), the same ranker in **TensorFlow/Keras**, and **visual search** with a
pretrained CNN (computer vision) on real images.

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
- **Limit:** pin titles here only reveal the topic. With real text or image embeddings (see visual search below)
  the content model, and cold-start recall, should do much better.

## Same ranker in TensorFlow / Keras

`pinrec/tf_ranker.py` rebuilds the stage-2 ranker in Keras (same features, layers and class weighting).
Recall@20 0.188 vs 0.189 in PyTorch, NDCG@20 0.153 in both: the gain comes from the features, not the framework.

## Ads ranking: CTR prediction, calibration and auction (`pinrec/ads.py`)

Each ad promotes a pin and has a cost-per-click bid. Generated impressions: the true click chance depends on topic
match, a hidden ad quality and the slot position. Models see only realistic signals: topic affinity from past saves,
the LightGCN score, the ad's earlier click history (shrunk toward the average) and position.

| CTR model | AUC | Log loss | Calibration error (ECE) |
|---|---|---|---|
| Average CTR (baseline) | 0.500 | 0.0842 | 0.0004 |
| Logistic regression (PyTorch) | 0.778 | 0.0754 | 0.0043 |
| Neural network (PyTorch) | **0.801** | **0.0720** | **0.0024** |

| Auction rule (1 slot, 30 candidate ads) | Clicks / 1,000 requests | Revenue / 1,000 requests (CAD) | Ad matches user interests |
|---|---|---|---|
| Bid only | 19.8 | 50.1 | 10% |
| Predicted CTR only | 163.2 | 119.7 | 86% |
| **Bid × predicted CTR** | 134.6 | **143.5** | 77% |

- **Position bias:** CTR is 2.47% in slot 1 and 1.14% in slot 4 for the same pool of ads, so the model learns position
  and scores every ad at slot 1 when ranking.
- **Calibration matters:** the auction multiplies bids by predicted CTR, so the model must be calibrated, not just rank
  well. No class weighting is used here, unlike the pin ranker.
- **Trade-off:** bid × predicted CTR earns almost 3 times more than bid alone, and still shows a relevant ad 77% of the time.
  Revenue uses bid × true click chance, a simplification of a real second-price auction.

## Visual search: computer vision with a pretrained CNN (`scripts/visual_search.py`)

"More like this" for images: embed each image with ResNet-18 (penultimate layer, 512 dimensions) and return the
nearest neighbours by cosine similarity. Real images: 2,000 from the CIFAR-10 test set (10 classes). A result is
correct if it shares the query's class. Compared with raw pixels and with the same network with random weights.
It downloads images and weights, so it runs in the `visual-search` GitHub Actions workflow; results are in
[`results/VISUAL_SEARCH.md`](results/VISUAL_SEARCH.md).

## MLflow

Every run logs its parameters (model, embedding size, layers, epochs, learning rate, candidate counts, features)
and metrics (recall, NDCG, cold-start recall, coverage, training time) to `mlflow.db`.

```
pip install mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 and compare the runs side by side (recommendation models, the TensorFlow ranker and the 3 CTR models).

## Run it (Windows Command Prompt)

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install tensorflow-cpu torchvision
pytest -q
python -m pinrec
```

`pytest -q` shows **15 passed**. `python -m pinrec` trains every model on CPU in about a minute and writes
`results/RESULTS.md`, `results/metrics.csv` and `mlflow.db`. Small run-to-run differences (third decimal) come
from multi-threaded CPU math.

## Tests

- Split is temporal; new pins appear only in the test period.
- The graph adjacency is symmetric and normalized as D^-1/2 A D^-1/2.
- LightGCN layers mix in neighbour embeddings; BPR loss falls during training.
- top-k never repeats a pin the user already saved; recall, NDCG, cold-start recall and coverage match hand-computed values.
- Fresh candidates only add unsaved pins; fresh slots work as described.
- End-to-end run on a small dataset beats popularity and logs to MLflow.
- Ads: position bias and relevance in the click model; AUC and calibration error match hand-computed values; the
  CTR model is calibrated (ECE < 0.01); bid × predicted CTR beats bid-only on revenue and relevance.
- The Keras ranker learns the planted signal; visual search precision@k is 1.0 for perfectly separated embeddings.

Visual search itself: `python scripts/visual_search.py` (needs internet the first time).

## Files

| File | What it does |
|---|---|
| `pinrec/data.py` | Generates users, pins (topic, title, new flag) and saves; temporal split |
| `pinrec/models.py` | Matrix factorization, LightGCN, BPR training, text embeddings |
| `pinrec/rank.py` | Candidate generation, ranking features, PyTorch MLP ranker, fresh slots |
| `pinrec/evaluate.py` | top-k, recall@k, NDCG@k, cold-start recall, catalog coverage |
| `pinrec/ads.py` | Ad impressions, CTR models, calibration, position bias, auction |
| `pinrec/tf_ranker.py` | The stage-2 ranker in TensorFlow / Keras |
| `scripts/visual_search.py` | Visual search with pretrained ResNet-18 on CIFAR-10 |
| `pinrec/__main__.py` | Runs every experiment, logs to MLflow, writes results |
