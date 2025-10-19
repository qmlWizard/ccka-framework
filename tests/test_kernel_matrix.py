import numpy as np
from backend.matrix.full_matrix import full_matrix


def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 0.5) -> float:
    """Radial Basis Function (Gaussian) kernel."""
    diff = x - y
    return np.exp(-gamma * np.dot(diff, diff))


def test_full_matrix_shape_and_values():
    """
    Test that full_matrix computes a valid RBF kernel matrix
    with correct dimensions and finite numerical values.
    """
    np.random.seed(42)
    X = np.random.randn(50, 4)
    Y = np.random.randn(60, 4)

    K = full_matrix(rbf_kernel, X, Y)

    # ---- Assertions ------------------------------------------------------
    assert isinstance(K, np.ndarray), "Output must be a NumPy array."
    assert K.shape == (50, 60), f"Unexpected kernel matrix shape: {K.shape}"
    assert np.isfinite(K).all(), "Kernel matrix contains non-finite values."
    assert (K >= 0).all() and (K <= 1).all(), "RBF kernel values must be in [0, 1]."

    # Sanity check: self-similarity should be maximum
    idx = np.random.randint(0, 10)
    self_sim = rbf_kernel(X[idx], X[idx])
    assert np.isclose(self_sim, 1.0, atol=1e-8), "RBF self-similarity must be 1."


def test_full_matrix_thread_consistency():
    """
    Test that multi-threaded execution yields deterministic results.
    """
    np.random.seed(0)
    X = np.random.randn(20, 3)
    Y = np.random.randn(25, 3)

    K1 = full_matrix(rbf_kernel, X, Y)
    K2 = full_matrix(rbf_kernel, X, Y)

    assert np.allclose(K1, K2, atol=1e-10), "Results must be deterministic across runs."
