import numpy as np
from .base_kernel import BaseKernel
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel, cosine_similarity


class RBFKernel(BaseKernel):
    """Radial Basis Function (Gaussian) kernel."""
    def __init__(self, gamma: float = 0.5, seed: int | None = None):
        super().__init__(name="RBF", params={"gamma": gamma}, seed=seed)
        self.gamma = gamma

    def compute(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        self.validate_inputs(X, Y)
        return rbf_kernel(X, Y, gamma=self.gamma)


class PolynomialKernel(BaseKernel):
    """Polynomial kernel: (γ⟨x, y⟩ + coef0)^degree"""
    def __init__(self, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0, seed: int | None = None):
        super().__init__("Polynomial", {"degree": degree, "gamma": gamma, "coef0": coef0}, seed)
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def compute(self, X, Y=None):
        self.validate_inputs(X, Y)
        return polynomial_kernel(X, Y, degree=self.degree, gamma=self.gamma, coef0=self.coef0)


class LinearKernel(BaseKernel):
    """Linear kernel: ⟨x, y⟩"""
    def __init__(self, seed: int | None = None):
        super().__init__("Linear", {}, seed)

    def compute(self, X, Y=None):
        self.validate_inputs(X, Y)
        return linear_kernel(X, Y)


class CosineKernel(BaseKernel):
    """Cosine similarity kernel."""
    def __init__(self, seed: int | None = None):
        super().__init__("Cosine", {}, seed)

    def compute(self, X, Y=None):
        self.validate_inputs(X, Y)
        return cosine_similarity(X, Y)
