import numpy as np
from backend.metrics.kta import kta, center_kernel


def test_kta_range():
    X = np.random.randn(5, 5)
    K1 = X @ X.T
    K2 = K1 + 0.01 * np.random.randn(5, 5)
    score = kta(K1, K2)
    assert -1 <= score <= 1


def test_kta_centering_invariance():
    X = np.random.randn(4, 4)
    K = X @ X.T
    Kc = center_kernel(K)
    assert np.allclose(center_kernel(Kc), Kc)  # idempotent centering
