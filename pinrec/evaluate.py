"""Offline ranking metrics: recall@k, NDCG@k, cold-start recall and catalog coverage."""
from __future__ import annotations

import numpy as np
import pandas as pd


def top_k(scores: np.ndarray, seen: dict[int, np.ndarray], k: int) -> np.ndarray:
    """Top-k pins per user, never recommending a pin the user already saved."""
    s = scores.copy()
    for u, items in seen.items():
        s[u, items] = -np.inf
    part = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.take_along_axis(s, part, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(part, order, 1)


def seen_items(*frames: pd.DataFrame) -> dict[int, np.ndarray]:
    df = pd.concat(frames)
    return {u: g.to_numpy() for u, g in df.groupby("user_id")["pin_id"]}


def metrics(recs: np.ndarray, truth: pd.DataFrame, new_pins: set[int], n_pins: int) -> dict:
    k = recs.shape[1]
    t = {u: set(g) for u, g in truth.groupby("user_id")["pin_id"]}
    disc = 1 / np.log2(np.arange(2, k + 2))
    rec, ndcg, cold_hit, cold_tot = [], [], 0, 0
    for u, items in t.items():
        r = recs[u]
        hits = np.array([p in items for p in r])
        rec.append(hits.sum() / len(items))
        ideal = disc[: min(len(items), k)].sum()
        ndcg.append((hits * disc).sum() / ideal)
        new_true = items & new_pins
        if new_true:
            cold_tot += len(new_true)
            cold_hit += len(new_true & set(r))
    return {
        f"recall_at_{k}": float(np.mean(rec)),
        f"ndcg_at_{k}": float(np.mean(ndcg)),
        f"cold_start_recall_at_{k}": cold_hit / cold_tot if cold_tot else 0.0,
        "catalog_coverage": len(np.unique(recs)) / n_pins,
    }
