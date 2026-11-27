"""Run every experiment, track it in MLflow and write results/RESULTS.md: python -m pinrec"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)

from . import data, models, rank
from . import evaluate as ev

ROOT = Path(__file__).resolve().parent.parent
K = 20


def scores(model) -> np.ndarray:
    with torch.no_grad():
        u, p = model.embeddings()
        return (u @ p.T).numpy()


def run(out: Path = ROOT / "results", tracking: Path = ROOT / "mlflow.db", epochs: int = 20, small: bool = False) -> dict:
    out.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking.resolve()}")
    mlflow.set_experiment("pin-recommendations")
    ds = data.generate(n_users=800, n_pins=1600) if small else data.generate()
    # Hide a random 8% of older pins from the graph's training saves. In the validation period they look like
    # brand-new pins, so the ranker learns how to rank pins that have no saves yet (it never sees real new pins).
    rng = np.random.default_rng(1)
    old = ds.pins.loc[~ds.pins["is_new"], "pin_id"].to_numpy()
    hidden = rng.choice(old, int(0.08 * len(old)), replace=False)
    ds.train = ds.train[~ds.train["pin_id"].isin(hidden)].reset_index(drop=True)
    new = set(ds.pins.loc[ds.pins["is_new"], "pin_id"])
    hist = pd.concat([ds.train, ds.valid])
    seen_test = ev.seen_items(ds.train, ds.valid)
    seen_valid = ev.seen_items(ds.train)
    pop = np.bincount(ds.train["pin_id"], minlength=ds.n_pins).astype(float)
    results = {}

    def log(name, params, m, secs):
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_metrics({**m, "train_seconds": secs})
        results[name] = {**m, "train_seconds": secs}

    # 1. popularity baseline
    log("popularity", {"model": "popularity"}, ev.metrics(ev.top_k(np.tile(pop, (ds.n_users, 1)), seen_test, K), ds.test, new, ds.n_pins), 0.0)

    # 2. matrix factorization (BPR)
    t = time.time()
    torch.manual_seed(0)
    mf = models.MF(ds.n_users, ds.n_pins)
    mf_loss = models.train_bpr(mf, ds, epochs=epochs, lr=0.005)
    log("matrix_factorization", {"model": "MF", "dim": 64, "epochs": epochs, "lr": 0.005, "loss": "BPR"},
        ev.metrics(ev.top_k(scores(mf), seen_test, K), ds.test, new, ds.n_pins), round(time.time() - t, 1))

    # 3. LightGCN (graph-based)
    t = time.time()
    adj = models.normalized_adjacency(ds)
    torch.manual_seed(0)
    lg = models.LightGCN(ds.n_users, ds.n_pins, adj, layers=2)
    lg_loss = models.train_bpr(lg, ds, epochs=epochs, lr=0.005)
    graph = scores(lg)
    log("lightgcn", {"model": "LightGCN", "dim": 64, "layers": 2, "epochs": epochs, "lr": 0.005, "loss": "BPR"},
        ev.metrics(ev.top_k(graph, seen_test, K), ds.test, new, ds.n_pins), round(time.time() - t, 1))

    # 4. content model (pin title text embeddings)
    ce = models.content_embeddings(ds)
    content_valid = models.user_content_profiles(ds, ce, ds.train) @ ce.T
    content_test = models.user_content_profiles(ds, ce, hist) @ ce.T
    log("content_text", {"model": "TF-IDF+SVD content", "dim": 32},
        ev.metrics(ev.top_k(content_test, seen_test, K), ds.test, new, ds.n_pins), 0.0)

    # 5. two-stage: LightGCN + content candidates, neural ranker trained on the validation period
    t = time.time()
    n_g, n_c = 200, 50
    fresh = pop == 0
    cv = rank.candidates(graph, content_valid, seen_valid, n_g, n_c, fresh)
    fv = rank.features(cv, graph, content_valid, ds.train, ds.pins, pop)
    model, norm, pos_rate = rank.fit_ranker(fv, ds.valid)
    ct = rank.candidates(graph, content_test, seen_test, n_g, n_c, fresh)
    ft = rank.features(ct, graph, content_test, hist, ds.pins, pop)
    cand_recall = ft.merge(ds.test, on=["user_id", "pin_id"]).shape[0] / len(ds.test)
    recs = rank.rank(model, norm, ft, ds.n_users, K)
    m = ev.metrics(recs, ds.test, new, ds.n_pins)
    m["candidate_recall"] = cand_recall
    log("two_stage_ranker", {"model": "LightGCN + fresh-pin content candidates -> MLP ranker", "graph_candidates": n_g,
                             "content_candidates": n_c, "features": ",".join(rank.FEATURES)}, m, round(time.time() - t, 1))

    # ablation: ranker without content candidates (graph only)
    g_top = ev.top_k(graph, seen_test, n_g)
    ct2 = [c[np.isin(c, g_top[u])] for u, c in enumerate(ct)]
    ft2 = ft.merge(pd.DataFrame({"user_id": np.repeat(np.arange(ds.n_users), [len(c) for c in ct2]),
                                 "pin_id": np.concatenate(ct2)}), on=["user_id", "pin_id"])
    m2 = ev.metrics(rank.rank(model, norm, ft2, ds.n_users, K), ds.test, new, ds.n_pins)
    log("two_stage_graph_candidates_only", {"model": "ablation: no content candidates"}, m2, 0.0)

    m3 = ev.metrics(rank.rank(model, norm, ft, ds.n_users, K, fresh_slots=3), ds.test, new, ds.n_pins)
    m3["candidate_recall"] = cand_recall
    log("two_stage_ranker_fresh_slots", {"model": "two-stage ranker + 3 of 20 slots for fresh pins", "fresh_slots": 3}, m3, 0.0)

    write(out, ds, results, mf_loss, lg_loss, pos_rate)
    return results


def write(out: Path, ds, r: dict, mf_loss, lg_loss, pos_rate) -> None:
    order = ["popularity", "matrix_factorization", "lightgcn", "content_text", "two_stage_graph_candidates_only", "two_stage_ranker", "two_stage_ranker_fresh_slots"]
    label = {"popularity": "Popularity baseline", "matrix_factorization": "Matrix factorization (BPR)",
             "lightgcn": "LightGCN, 2 layers (graph-based)", "content_text": "Content: pin title embeddings",
             "two_stage_graph_candidates_only": "Two-stage, graph candidates only (ablation)",
             "two_stage_ranker": "Two-stage: graph + fresh-pin candidates → neural ranker",
             "two_stage_ranker_fresh_slots": "**Two-stage ranker + 3 of 20 slots for fresh pins**"}
    rows = "\n".join(
        f"| {label[k]} | {r[k][f'recall_at_{K}']:.3f} | {r[k][f'ndcg_at_{K}']:.3f} | {r[k][f'cold_start_recall_at_{K}']:.3f} | {r[k]['catalog_coverage']:.1%} |"
        for k in order)
    two, lg = r["two_stage_ranker_fresh_slots"], r["lightgcn"]
    text = f"""# Results (generated sample data)

{ds.n_users:,} users, {ds.n_pins:,} pins in 24 topics, {len(ds.train) + len(ds.valid) + len(ds.test):,} saves.
Temporal split per user: first 70% train, next 10% validation (trains the ranker), last 20% test.\nAnother 8% of older pins are hidden from the graph during training so the ranker can learn to rank pins with no saves.
{ds.pins['is_new'].mean():.0%} of pins are new (no saves before the test period); they are {ds.test['pin_id'].isin(ds.pins.loc[ds.pins['is_new'], 'pin_id']).mean():.0%} of test saves.

| Model | Recall@{K} | NDCG@{K} | Cold-start recall@{K} | Catalog coverage |
|---|---|---|---|---|
{rows}

- Graph propagation helps: LightGCN beats plain matrix factorization (recall@{K} {lg[f'recall_at_{K}']:.3f} vs {r['matrix_factorization'][f'recall_at_{K}']:.3f}).
- The two-stage system (LightGCN + fresh-pin candidates, then a neural ranker with graph, text, topic and popularity
  features) beats LightGCN alone: recall@{K} {r['two_stage_ranker'][f'recall_at_{K}']:.3f} vs {lg[f'recall_at_{K}']:.3f}, NDCG@{K} {r['two_stage_ranker'][f'ndcg_at_{K}']:.3f} vs {lg[f'ndcg_at_{K}']:.3f}.
- Graph-only models never recommend a new pin (cold-start recall 0), and even the ranker keeps new pins out of the
  top {K} because established pins score higher. Reserving 3 of {K} slots for the best fresh pins raises cold-start
  recall to {two[f'cold_start_recall_at_{K}']:.3f} and catalog coverage from {r['two_stage_ranker']['catalog_coverage']:.1%} to {two['catalog_coverage']:.1%}, for a small cost in
  overall recall ({r['two_stage_ranker'][f'recall_at_{K}']:.3f} → {two[f'recall_at_{K}']:.3f}). That is a product decision (fresh content vs short-term accuracy),
  best settled with an online A/B test.
- Limit: pin titles here only reveal the topic, so the content model cannot tell which new pin in a topic a user
  will like; with real text or image embeddings, cold-start recall should be higher.
- Stage 1 keeps {two['candidate_recall']:.1%} of test saves among about 250 candidates per user; stage 2 orders them.
- Ranker training data: {pos_rate:.2%} of validation candidates were positives (class weighting used).
- BPR loss, last epoch: MF {mf_loss[-1]:.3f}, LightGCN {lg_loss[-1]:.3f}.

Every run is tracked in MLflow (`mlflow.db`; open with `mlflow ui --backend-store-uri sqlite:///mlflow.db`): parameters, metrics and training time.
"""
    (out / "RESULTS.md").write_text(text, encoding="utf-8")
    pd.DataFrame(r).T.to_csv(out / "metrics.csv")


if __name__ == "__main__":
    run()
    print((ROOT / "results" / "RESULTS.md").read_text(encoding="utf-8"))
