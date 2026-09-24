"""Hermetic tests for the cross-disease unified graph (synthetic data only)."""
import numpy as np
import pytest

from diagbench.data.base import TabularDataset
from diagbench import xgraph


def _toy_dataset(name, n, d, shift, seed):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2), dtype=np.int64)
    X = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    X[y == 1] += shift  # separable classes
    return TabularDataset(name=name, X=X, y=y,
                          feature_names=[f"f{i}" for i in range(d)],
                          positive_label="pos", source_url="synthetic",
                          citation="toy")


@pytest.fixture
def toy():
    dss = [_toy_dataset("a", 60, 5, 2.0, 1),
           _toy_dataset("b", 60, 7, 2.0, 2),
           _toy_dataset("c", 60, 4, 2.0, 3)]
    idx = []
    for ds in dss:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(ds.y))
        idx.append((perm[:30], perm[30:42], perm[42:]))
    return dss, [i[0] for i in idx], [i[1] for i in idx], [i[2] for i in idx]


def test_encoder_output_shape(toy):
    dss, itr, iva, ite = toy
    encs, heads, Z = xgraph.train_encoders(dss, itr, iva, d_lat=6,
                                           epochs=30, patience=10)
    assert len(encs) == 3 and len(Z) == 3
    assert Z[0].shape == (60, 6) and Z[2].shape == (60, 6)


def test_unified_graph_blocks_and_ablation(toy):
    dss, itr, iva, ite = toy
    _, _, Z = xgraph.train_encoders(dss, itr, iva, epochs=20, patience=10)
    metas = [xgraph.meta_fingerprint(ds.X) for ds in dss]
    assert metas[0].shape == (60, 10)
    Zall, A, A_cross, block = xgraph.build_unified_graph(Z, metas, k=5)
    assert Zall.shape == (180, Z[0].shape[1] + 10)
    assert A.shape == (180, 180)
    # every cross edge sits between different blocks
    ii, jj = np.nonzero(A_cross)
    assert (block[ii] != block[jj]).all()
    # ablation removes exactly the cross-disease edges
    _, A_abl, _, _ = xgraph.build_unified_graph(Z, metas, k=5,
                                                keep_cross_disease=False)
    ii2, jj2 = np.nonzero(A_abl)
    assert (block[ii2] == block[jj2]).all()


def test_multitask_gcn_learns_separable(toy):
    dss, itr, iva, ite = toy
    _, _, Z = xgraph.train_encoders(dss, itr, iva, epochs=40, patience=10)
    metas = [xgraph.meta_fingerprint(ds.X) for ds in dss]
    Zall, A, _, block = xgraph.build_unified_graph(Z, metas, k=5)
    ys = [ds.y for ds in dss]
    model = xgraph.train_multitask_gcn(Zall, A, block, ys, itr, iva,
                                       epochs=60, patience=15)
    out = xgraph.evaluate_per_disease(model, Zall, A, block, ys, ite)
    assert set(out) == {0, 1, 2}
    for d, m in out.items():
        assert m["roc_auc"] > 0.9, (d, m)


def test_bridge_edges_labels_valid(toy):
    dss, itr, iva, ite = toy
    _, _, Z = xgraph.train_encoders(dss, itr, iva, epochs=20, patience=10)
    metas = [xgraph.meta_fingerprint(ds.X) for ds in dss]
    _, _, A_cross, block = xgraph.build_unified_graph(Z, metas, k=5)
    edges = xgraph.bridge_edges(A_cross, block, [ds.y for ds in dss])
    for e in edges:
        assert len(set(e["diseases"])) == 2
        assert e["labels"][0] in (0, 1) and e["labels"][1] in (0, 1)
        assert 0.0 < e["weight"] <= 1.0


def test_summarize_bridges_recurrence():
    seeds = [[{"diseases": [0, 1], "labels": (1, 1), "weight": 0.5}]
             for _ in range(5)]
    seeds[0] = seeds[0] + [{"diseases": [1, 2], "labels": (0, 1),
                            "weight": 0.3}]
    stable = xgraph.summarize_bridges(seeds, ["a", "b", "c"],
                                      min_seed_recurrence=4)
    assert len(stable) == 1 and stable[0]["pair"] == "a<->b"
    assert stable[0]["seed_recurrence"] == 5
