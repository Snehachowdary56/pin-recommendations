import numpy as np
import pandas as pd
import pytest
import torch

from pinrec import data, models, rank
from pinrec import evaluate as ev
from pinrec.__main__ import run


@pytest.fixture(scope="module")
def ds():
    return data.generate(n_users=300, n_pins=600, seed=3)


def test_split_is_temporal_and_new_pins_only_in_test(ds):
    new = set(ds.pins.loc[ds.pins["is_new"], "pin_id"])
    assert not ds.train["pin_id"].isin(new).any()
    assert not ds.valid["pin_id"].isin(new).any()
    assert ds.test["pin_id"].isin(new).any()
    m = ds.train.groupby("user_id")["t"].max()
    t = ds.test.groupby("user_id")["t"].min()
    both = m.index.intersection(t.index)
    assert (m[both] < t[both]).all()


def test_adjacency_is_symmetric_and_normalized(ds):
    a = models.normalized_adjacency(ds).to_dense()
    assert torch.allclose(a, a.T)
    u, p = int(ds.train["user_id"].iloc[0]), int(ds.train["pin_id"].iloc[0]) + ds.n_users
    du = (ds.train["user_id"] == u).sum()
    dp = (ds.train["pin_id"] == p - ds.n_users).sum()
    assert float(a[u, p]) == pytest.approx(1 / np.sqrt(du * dp), rel=1e-5)


def test_lightgcn_propagates_neighbours(ds):
    torch.manual_seed(0)
    m = models.LightGCN(ds.n_users, ds.n_pins, models.normalized_adjacency(ds), layers=2)
    u, _ = m.embeddings()
    assert not torch.allclose(u, m.emb.weight[: ds.n_users])  # layers mix in neighbour embeddings


def test_bpr_loss_falls(ds):
    torch.manual_seed(0)
    m = models.MF(ds.n_users, ds.n_pins, dim=16)
    losses = models.train_bpr(m, ds, epochs=5, lr=0.01)
    assert losses[-1] < losses[0]


def test_top_k_never_repeats_saved_pins():
    scores = np.array([[5.0, 4, 3, 2, 1], [1, 2, 3, 4, 5]])
    recs = ev.top_k(scores, {0: np.array([0]), 1: np.array([4, 3])}, 2)
    assert recs.tolist() == [[1, 2], [2, 1]]


def test_metrics_known_values():
    recs = np.array([[1, 2, 3]])
    truth = pd.DataFrame({"user_id": [0, 0], "pin_id": [1, 9]})
    m = ev.metrics(recs, truth, new_pins={9}, n_pins=10)
    assert m["recall_at_3"] == 0.5
    assert m["ndcg_at_3"] == pytest.approx(1 / (1 + 1 / np.log2(3)))
    assert m["cold_start_recall_at_3"] == 0.0
    assert m["catalog_coverage"] == 0.3


def test_content_embeddings_cover_new_pins(ds):
    ce = models.content_embeddings(ds)
    assert ce.shape[0] == ds.n_pins
    assert np.allclose(np.linalg.norm(ce, axis=1), 1, atol=1e-6)


def test_fresh_candidates_only_add_unsaved_pins():
    graph = np.array([[3.0, 2, 1, 0]])
    content = np.array([[0.0, 0, 1, 2]])
    fresh = np.array([False, False, False, True])
    c = rank.candidates(graph, content, {}, 2, 1, fresh)
    assert sorted(c[0].tolist()) == [0, 1, 3]


def test_fresh_slots_reserve_new_pins():
    df = pd.DataFrame({"user_id": 0, "pin_id": [0, 1, 2, 3], **{f: 0.0 for f in rank.FEATURES}})
    df["graph_score"] = [4.0, 3, 2, 1]
    df["is_new_pin"] = [0.0, 0, 0, 1]

    class Score(torch.nn.Module):
        def forward(self, x):
            return x[:, 0]
    norm = (np.zeros(len(rank.FEATURES), dtype=np.float32), np.ones(len(rank.FEATURES), dtype=np.float32))
    assert rank.rank(Score(), norm, df, 1, 2).tolist() == [[0, 1]]
    assert rank.rank(Score(), norm, df, 1, 2, fresh_slots=1).tolist() == [[0, 3]]


def test_end_to_end_small(tmp_path):
    r = run(out=tmp_path / "results", tracking=tmp_path / "mlflow.db", epochs=8, small=True)
    assert r["lightgcn"]["recall_at_20"] > r["popularity"]["recall_at_20"]
    assert r["two_stage_ranker_fresh_slots"]["cold_start_recall_at_20"] > 0
    assert r["lightgcn"]["cold_start_recall_at_20"] == 0
    assert (tmp_path / "results" / "RESULTS.md").exists()
    assert (tmp_path / "mlflow.db").exists()
