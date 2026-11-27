"""Candidate generators in PyTorch: matrix factorization (BPR), LightGCN (graph-based), and a content model."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn

from .data import Dataset


def _bpr_batches(train, n_pins: int, batch: int, rng: np.random.Generator):
    u = train["user_id"].to_numpy()
    p = train["pin_id"].to_numpy()
    idx = rng.permutation(len(u))
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        neg = rng.integers(0, n_pins, len(b))
        yield torch.from_numpy(u[b]), torch.from_numpy(p[b]), torch.from_numpy(neg)


class MF(nn.Module):
    """Matrix factorization: score = user embedding . pin embedding."""

    def __init__(self, n_users: int, n_pins: int, dim: int = 64):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.pin = nn.Embedding(n_pins, dim)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.pin.weight, std=0.1)

    def embeddings(self):
        return self.user.weight, self.pin.weight


class LightGCN(nn.Module):
    """Graph-based embeddings: propagate ID embeddings over the normalized user-pin graph for K layers
    and average the layers (He et al., 2020). Neighbours' tastes flow into each node's embedding."""

    def __init__(self, n_users: int, n_pins: int, adj: torch.Tensor, dim: int = 64, layers: int = 2):
        super().__init__()
        self.n_users, self.layers, self.adj = n_users, layers, adj
        self.emb = nn.Embedding(n_users + n_pins, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def embeddings(self):
        x = self.emb.weight
        out = [x]
        for _ in range(self.layers):
            x = torch.sparse.mm(self.adj, x)
            out.append(x)
        h = torch.stack(out).mean(0)
        return h[: self.n_users], h[self.n_users:]


def normalized_adjacency(ds: Dataset) -> torch.Tensor:
    """Symmetric-normalized adjacency D^-1/2 A D^-1/2 of the bipartite user-pin graph."""
    n = ds.n_users + ds.n_pins
    u = ds.train["user_id"].to_numpy()
    p = ds.train["pin_id"].to_numpy() + ds.n_users
    rows = np.concatenate([u, p])
    cols = np.concatenate([p, u])
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    dinv = np.zeros_like(deg)
    dinv[deg > 0] = deg[deg > 0] ** -0.5
    vals = dinv[rows] * dinv[cols]
    return torch.sparse_coo_tensor(torch.tensor(np.vstack([rows, cols])), torch.tensor(vals, dtype=torch.float32), (n, n)).coalesce()


def train_bpr(model: nn.Module, ds: Dataset, epochs: int = 20, lr: float = 0.01, batch: int = 4096,
              reg: float = 1e-4, seed: int = 0, log=None) -> list[float]:
    """Bayesian personalized ranking: a saved pin should score above a random pin."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for ep in range(epochs):
        total, n = 0.0, 0
        for u, p, q in _bpr_batches(ds.train, ds.n_pins, batch, rng):
            ue, pe = model.embeddings()
            su = ue[u]
            pos = (su * pe[p]).sum(1)
            neg = (su * pe[q]).sum(1)
            loss = -torch.nn.functional.logsigmoid(pos - neg).mean() + reg * (su.pow(2).sum(1) + pe[p].pow(2).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(u)
            n += len(u)
        losses.append(total / n)
        if log:
            log(ep, losses[-1])
    return losses


def content_embeddings(ds: Dataset, dim: int = 32, seed: int = 0) -> np.ndarray:
    """Text embeddings of pin titles (TF-IDF + SVD), L2-normalized. Works for brand-new pins."""
    tf = TfidfVectorizer().fit_transform(ds.pins["title"])
    z = TruncatedSVD(dim, random_state=seed).fit_transform(tf)
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)


def user_content_profiles(ds: Dataset, pin_emb: np.ndarray, history) -> np.ndarray:
    """A user's content profile = mean embedding of the pins they saved."""
    prof = np.zeros((ds.n_users, pin_emb.shape[1]))
    np.add.at(prof, history["user_id"].to_numpy(), pin_emb[history["pin_id"].to_numpy()])
    return prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9)
