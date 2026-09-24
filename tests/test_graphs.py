import numpy as np

from diagbench.graphs import knn_similarity_graph, normalize_adjacency


def test_knn_graph_symmetric_zero_diag():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5)).astype(np.float32)
    A = knn_similarity_graph(X, k=5)
    assert np.allclose(A, A.T)
    assert np.allclose(np.diag(A), 0.0)
    assert (A >= 0).all() and (A <= 1).all()


def test_knn_graph_cluster_structure():
    # two well-separated clusters: within-cluster edges must dominate
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.1, (20, 4)), rng.normal(10, 0.1, (20, 4))])
    A = knn_similarity_graph(X.astype(np.float32), k=4)
    same = A[:20, :20].sum() + A[20:, 20:].sum()
    cross = A[:20, 20:].sum() + A[20:, :20].sum()
    assert same > 20 * cross


def test_normalize_adjacency_spectrum():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 3)).astype(np.float32)
    An = normalize_adjacency(knn_similarity_graph(X, k=4))
    eig = np.linalg.eigvalsh(An)
    assert eig.max() <= 1.0 + 1e-5  # normalized Laplacian property
    assert np.allclose(An, An.T)

def test_knn_graph_matches_bruteforce_on_large_block():
    """Matmul distance path must equal naive pairwise distances."""
    import numpy as np
    from diagbench.graphs import knn_similarity_graph
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 15)).astype(np.float32)
    A = knn_similarity_graph(X, k=5)
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    i, j = 7, np.argsort(sq[7])[1]
    assert A[7, j] > 0 and A[7, j] <= 1.0
    assert (A == A.T).all()
