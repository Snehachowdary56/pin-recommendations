"""Ads ranking: predict click-through rate (CTR), check calibration, and rank ads in an auction.

Generated sample data, not real ad data. Each ad promotes a pin and has a cost-per-click bid. The true click
chance depends on how well the ad's topic matches the user's interests, a hidden ad quality, and the slot
position (people click lower slots less: position bias). Bids do not change click chance.

The models only see what a real system would: the user's topic affinity from past saves, the LightGCN
user-pin score, the ad's click history from an earlier period, and the slot position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

from .data import TOPIC_NAMES, Dataset

FEATURES = ["topic_affinity", "graph_score", "ad_hist_ctr", "position"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def true_ctr(rel: np.ndarray, quality: np.ndarray, position: np.ndarray) -> np.ndarray:
    return _sigmoid(-4.6 + 2.4 * rel + 0.9 * quality - 0.35 * (position - 1))


def make_ads(ds: Dataset, n_ads: int = 400, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    old = ds.pins.loc[~ds.pins["is_new"], "pin_id"].to_numpy()
    pins = rng.choice(old, n_ads, replace=False)
    return pd.DataFrame({
        "ad_id": np.arange(n_ads), "pin_id": pins,
        "topic": ds.pins.set_index("pin_id").loc[pins, "topic"].to_numpy(),
        "bid": np.round(np.exp(rng.normal(np.log(0.6), 0.6, n_ads)).clip(0.1, 4.0), 2),   # CAD per click
        "quality": rng.normal(0, 1, n_ads),   # hidden from the models
    })


def _relevance(ds: Dataset, users: np.ndarray, ad_topics: np.ndarray) -> np.ndarray:
    ints = ds.users.set_index("user_id")["interests"]
    return np.array([1.0 if t in ints[u] else 0.0 for u, t in zip(users, ad_topics)])


def topic_affinity(ds: Dataset, history: pd.DataFrame) -> np.ndarray:
    tix = {t: i for i, t in enumerate(TOPIC_NAMES)}
    pin_t = ds.pins["topic"].map(tix).to_numpy()
    aff = np.zeros((ds.n_users, len(TOPIC_NAMES)))
    np.add.at(aff, (history["user_id"].to_numpy(), pin_t[history["pin_id"].to_numpy()]), 1)
    return aff / np.maximum(aff.sum(1, keepdims=True), 1)


def impressions(ds: Dataset, ads: pd.DataFrame, graph: np.ndarray, history: pd.DataFrame,
                n: int = 120_000, seed: int = 22) -> pd.DataFrame:
    """Ad impressions over 3 periods: 'history' (gives each ad a click history), 'train' and 'test'."""
    rng = np.random.default_rng(seed)
    users = rng.integers(0, ds.n_users, n)
    a = rng.integers(0, len(ads), n)
    pos = rng.integers(1, 5, n)
    ad_topic = ads["topic"].to_numpy()[a]
    rel = _relevance(ds, users, ad_topic)
    p = true_ctr(rel, ads["quality"].to_numpy()[a], pos)
    click = rng.random(n) < p
    aff = topic_affinity(ds, history)
    tix = {t: i for i, t in enumerate(TOPIC_NAMES)}
    gs = graph[users, ads["pin_id"].to_numpy()[a]]
    period = np.where(np.arange(n) < n * 0.3, "history", np.where(np.arange(n) < n * 0.75, "train", "test"))
    df = pd.DataFrame({"user_id": users, "ad_id": a, "position": pos, "click": click.astype(int), "true_ctr": p,
                       "topic_affinity": aff[users, [tix[t] for t in ad_topic]],
                       "graph_score": (gs - gs.mean()) / gs.std(), "period": period})
    hist = df[df["period"] == "history"].groupby("ad_id")["click"].agg(["sum", "count"])
    prior = df.loc[df["period"] == "history", "click"].mean()
    smoothed = (hist["sum"] + 20 * prior) / (hist["count"] + 20)   # shrink small samples toward the average
    df["ad_hist_ctr"] = df["ad_id"].map(smoothed).fillna(prior)
    return df[df["period"] != "history"].reset_index(drop=True)


class CTRModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 0):
        super().__init__()
        self.net = nn.Linear(n_features, 1) if hidden == 0 else nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _xy(df, mu=None, sd=None):
    x = df[FEATURES].to_numpy(dtype=np.float32)
    x[:, FEATURES.index("ad_hist_ctr")] = np.log(x[:, FEATURES.index("ad_hist_ctr")])
    if mu is None:
        mu, sd = x.mean(0), x.std(0) + 1e-6
    return torch.tensor((x - mu) / sd), torch.tensor(df["click"].to_numpy(dtype=np.float32)), mu, sd


def fit_ctr(train: pd.DataFrame, hidden: int = 0, epochs: int = 12, lr: float = 3e-3, seed: int = 0):
    torch.manual_seed(seed)
    x, y, mu, sd = _xy(train)
    m = CTRModel(x.shape[1], hidden)
    last = m.net if hidden == 0 else m.net[-1]
    with torch.no_grad():   # start at the average click rate, so training refines rather than climbs down from 50%
        last.bias.fill_(float(np.log(y.mean() / (1 - y.mean()))))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()   # no class weights: we want calibrated probabilities
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for s in range(0, len(perm), 2048):
            b = perm[s:s + 2048]
            loss = lossf(m(x[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m, (mu, sd)


def predict(m, norm, df: pd.DataFrame, at_position: int | None = None) -> np.ndarray:
    d = df if at_position is None else df.assign(position=at_position)
    x, _, _, _ = _xy(d, *norm)
    with torch.no_grad():
        return torch.sigmoid(m(x)).numpy()


def ctr_metrics(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """AUC (ranking), log loss, and expected calibration error over predicted-CTR deciles."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop") if np.ptp(p) > 0 else np.zeros(len(p))
    t = pd.DataFrame({"y": y, "p": p, "b": q}).groupby("b").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    ece = float((t["n"] * (t["pred"] - t["actual"]).abs()).sum() / t["n"].sum())
    return {"auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)), "ece": ece,
            "mean_pred": float(p.mean()), "mean_actual": float(y.mean())}


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    return pd.DataFrame({"y": y, "p": p, "b": q}).groupby("b").agg(predicted_ctr=("p", "mean"), actual_ctr=("y", "mean"), impressions=("y", "size")).reset_index(drop=True)


def auction(ds: Dataset, ads: pd.DataFrame, m, norm, graph: np.ndarray, history: pd.DataFrame,
            n_requests: int = 3000, n_candidates: int = 30, seed: int = 23) -> pd.DataFrame:
    """One ad slot per request. Compare ranking ads by bid, by predicted CTR, and by expected value
    (bid x predicted CTR, often called eCPM). Outcomes use the TRUE click chance at slot 1."""
    rng = np.random.default_rng(seed)
    aff = topic_affinity(ds, history)
    tix = {t: i for i, t in enumerate(TOPIC_NAMES)}
    rows = []
    gs_all = graph[:, ads["pin_id"].to_numpy()]
    mu_g, sd_g = gs_all.mean(), gs_all.std()
    for r in range(n_requests):
        u = int(rng.integers(0, ds.n_users))
        c = rng.choice(len(ads), n_candidates, replace=False)
        sub = ads.iloc[c]
        feats = pd.DataFrame({
            "topic_affinity": aff[u, [tix[t] for t in sub["topic"]]],
            "graph_score": (gs_all[u, c] - mu_g) / sd_g,
            "ad_hist_ctr": sub["hist_ctr"].to_numpy(),
            "position": 1, "click": 0})
        pctr = predict(m, norm, feats)
        rel = _relevance(ds, np.full(len(c), u), sub["topic"].to_numpy())
        tctr = true_ctr(rel, sub["quality"].to_numpy(), np.ones(len(c)))
        bid = sub["bid"].to_numpy()
        for rule, score in (("bid only", bid), ("predicted CTR only", pctr), ("bid x predicted CTR", bid * pctr)):
            w = int(np.argmax(score))
            rows.append((rule, tctr[w], bid[w] * tctr[w], rel[w]))
    out = pd.DataFrame(rows, columns=["rule", "ctr", "revenue", "relevant"])
    g = out.groupby("rule", sort=False).agg(clicks_per_1000=("ctr", "mean"), revenue_per_1000=("revenue", "mean"),
                                            relevant_share=("relevant", "mean"))
    g["clicks_per_1000"] *= 1000
    g["revenue_per_1000"] *= 1000
    return g.reset_index()


def run(ds: Dataset, graph: np.ndarray, history: pd.DataFrame) -> dict:
    ads = make_ads(ds)
    imp = impressions(ds, ads, graph, history)
    ads["hist_ctr"] = ads["ad_id"].map(imp.groupby("ad_id")["ad_hist_ctr"].first()).fillna(imp["ad_hist_ctr"].mean())
    tr, te = imp[imp["period"] == "train"], imp[imp["period"] == "test"]
    y = te["click"].to_numpy()
    res = {"impressions_train": len(tr), "impressions_test": len(te), "base_ctr": float(tr["click"].mean())}
    res["average_ctr"] = ctr_metrics(y, np.full(len(y), tr["click"].mean()))
    lr, lr_norm = fit_ctr(tr, hidden=0)
    res["logistic_regression"] = ctr_metrics(y, predict(lr, lr_norm, te))
    mlp, mlp_norm = fit_ctr(tr, hidden=32)
    p_mlp = predict(mlp, mlp_norm, te)
    res["neural_net"] = ctr_metrics(y, p_mlp)
    res["reliability"] = reliability(y, p_mlp)
    # position bias: clicks fall with slot even for the same ads
    res["ctr_by_position"] = te.groupby("position")["click"].mean().to_dict()
    res["auction"] = auction(ds, ads, mlp, mlp_norm, graph, history)
    return res
