"""Two-stage recommendation: candidate generation, then a learned ranker (PyTorch MLP).

Stage 1 (retrieval): the top pins from LightGCN plus a "fresh pins" source (content model over pins with no saves
yet), so brand-new pins can still become candidates.
Stage 2 (ranking): a small neural network scores each (user, candidate) pair from features, trained on the
validation period (saves the candidate models never saw).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURES = ["graph_score", "content_score", "log_popularity", "is_new_pin", "topic_affinity", "graph_rank", "content_rank"]


def candidates(graph: np.ndarray, content: np.ndarray, seen: dict[int, np.ndarray], n_graph: int, n_content: int,
               fresh: np.ndarray | None = None) -> list[np.ndarray]:
    """Union of the graph model's top pins and a "fresh pins" source: the content model's top pins among pins
    with no saves yet (fresh = boolean mask over pins). The graph model cannot score fresh pins."""
    from .evaluate import top_k
    g = top_k(graph, seen, n_graph)
    if n_content == 0:
        return [np.unique(g[u]) for u in range(graph.shape[0])]
    c_scores = content if fresh is None else np.where(fresh[None, :], content, -np.inf)
    c = top_k(c_scores, seen, n_content)
    return [np.unique(np.concatenate([g[u], c[u][np.isfinite(c_scores[u, c[u]])]])) for u in range(graph.shape[0])]


def features(cands: list[np.ndarray], graph: np.ndarray, content: np.ndarray, history: pd.DataFrame,
             pins: pd.DataFrame, popularity: np.ndarray) -> pd.DataFrame:
    topic = pins["topic"].to_numpy()
    topics = sorted(set(topic))
    tix = {t: i for i, t in enumerate(topics)}
    pin_t = np.array([tix[t] for t in topic])
    aff = np.zeros((graph.shape[0], len(topics)))
    np.add.at(aff, (history["user_id"].to_numpy(), pin_t[history["pin_id"].to_numpy()]), 1)
    aff = aff / np.maximum(aff.sum(1, keepdims=True), 1)
    rows = []
    for u, c in enumerate(cands):
        gs, cs = graph[u, c], content[u, c]
        rows.append(pd.DataFrame({
            "user_id": u, "pin_id": c, "graph_score": gs, "content_score": cs,
            "log_popularity": np.log1p(popularity[c]), "is_new_pin": (popularity[c] == 0).astype(float),
            "topic_affinity": aff[u, pin_t[c]],
            "graph_rank": (-gs).argsort().argsort() / len(c), "content_rank": (-cs).argsort().argsort() / len(c),
        }))
    return pd.concat(rows, ignore_index=True)


class Ranker(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_ranker(df: pd.DataFrame, labels: pd.DataFrame, epochs: int = 8, lr: float = 3e-3, seed: int = 0):
    torch.manual_seed(seed)
    y = df.merge(labels.assign(y=1.0)[["user_id", "pin_id", "y"]], how="left", on=["user_id", "pin_id"])["y"].fillna(0).to_numpy()
    x = df[FEATURES].to_numpy(dtype=np.float32)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt = torch.tensor((x - mu) / sd)
    yt = torch.tensor(y, dtype=torch.float32)
    model = Ranker(len(FEATURES))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(xt), generator=g)
        for s in range(0, len(perm), 8192):
            b = perm[s:s + 8192]
            loss = lossf(model(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model, (mu, sd), float(y.mean())


def rank(model, norm, df: pd.DataFrame, n_users: int, k: int, fresh_slots: int = 0) -> np.ndarray:
    """Order each user's candidates by ranker score. With fresh_slots > 0, the best-scored pins with no saves
    yet fill that many of the k slots (exploration for new content); the rest go to the top overall."""
    mu, sd = norm
    with torch.no_grad():
        s = model(torch.tensor((df[FEATURES].to_numpy(dtype=np.float32) - mu) / sd)).numpy()
    df = df.assign(score=s).sort_values(["user_id", "score"], ascending=[True, False])
    out = np.zeros((n_users, k), dtype=int)
    for u, g in df.groupby("user_id"):
        ids = g["pin_id"].to_numpy()
        if fresh_slots:
            new = g["is_new_pin"].to_numpy() == 1
            fresh = ids[new][:fresh_slots]
            rest = [p for p in ids if p not in set(fresh)][: k - len(fresh)]
            ids = np.array(rest + list(fresh))
        out[u] = ids[:k]
    return out
