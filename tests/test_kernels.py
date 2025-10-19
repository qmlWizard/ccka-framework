"""
Unit tests for classical kernels.

To run:
    pytest -v tests/unit/test_classical_kernels.py
"""

import numpy as np
import pytest
from backend.kernels.classical import (
    RBFKernel,
    PolynomialKernel,
    LinearKernel,
    CosineKernel,
)


@pytest.fixture(scope="module")
def sample_data():
    """Fixture providing small reproducible datasets."""
    np.random.seed(42)
    X = np.random.randn(10, 3)
    Y = np.random.randn(8, 3)
    return X, Y


def _is_symmetric(K: np.ndarray) -> bool:
    return np.allclose(K, K.T, atol=1e-8)


def _is_psd(K: np.ndarray) -> bool:
    eigvals = np.linalg.eigvalsh(K)
    return np.all(eigvals >= -1e-8)


# --------------------------------------------------------------------------- #
#  RBF Kernel Tests
# --------------------------------------------------------------------------- #
def test_rbf_kernel_symmetry(sample_data):
    X, _ = sample_data
    kernel = RBFKernel(gamma=0.7)
    K = kernel.compute(X)
    assert _is_symmetric(K), "RBF kernel matrix must be symmetric."


def test_rbf_kernel_psd(sample_data):
    X, _ = sample_data
    kernel = RBFKernel(gamma=0.5)
    K = kernel.compute(X)
    assert _is_psd(K), "RBF kernel must be positive semidefinite."


def test_rbf_kernel_cross(sample_data):
    X, Y = sample_data
    kernel = RBFKernel(gamma=0.7)
    Kxy = kernel.compute(X, Y)
    assert Kxy.shape == (X.shape[0], Y.shape[0]), "Cross-kernel shape mismatch."


# --------------------------------------------------------------------------- #
#  Polynomial Kernel Tests
# --------------------------------------------------------------------------- #
def test_polynomial_kernel_basic(sample_data):
    X, _ = sample_data
    kernel = PolynomialKernel(degree=3, gamma=1.0, coef0=1.0)
    K = kernel.compute(X)
    assert _is_symmetric(K)
    assert _is_psd(K)


def test_polynomial_kernel_degree_effect(sample_data):
    X, _ = sample_data
    kernel_low = PolynomialKernel(degree=2)
    kernel_high = PolynomialKernel(degree=5)
    K_low = kernel_low.compute(X)
    K_high = kernel_high.compute(X)
    assert not np.allclose(K_low, K_high), "Different degrees should change kernel values."


# --------------------------------------------------------------------------- #
#  Linear & Cosine Kernel Tests
# --------------------------------------------------------------------------- #
def test_linear_kernel_symmetry(sample_data):
    X, _ = sample_data
    kernel = LinearKernel()
    K = kernel.compute(X)
    assert _is_symmetric(K)


def test_cosine_kernel_range(sample_data):
    X, _ = sample_data
    kernel = CosineKernel()
    K = kernel.compute(X)
    tol = 1e-6
    assert np.all((K >= -1 - tol) & (K <= 1 + tol)), (
        f"Cosine similarity values out of range "
        f"(min={K.min():.8f}, max={K.max():.8f})"
    )


# --------------------------------------------------------------------------- #
#  Parameter Handling & Config Tests
# --------------------------------------------------------------------------- #
def test_kernel_get_config_and_set_params():
    kernel = RBFKernel(gamma=0.5, seed=42)
    cfg = kernel.get_config()
    assert cfg["name"] == "RBF"
    assert "gamma" in cfg["params"]

    kernel.set_params(gamma=1.1)
    assert np.isclose(kernel.params["gamma"], 1.1)
    assert "RBF" in repr(kernel)
