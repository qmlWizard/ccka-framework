from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseKernel(ABC):
    """
    Abstract base class for all kernel types (classical and quantum).
    Provides standardized parameter handling, validation,
    and reproducible initialization.
    """

    def __init__(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        self.name = name
        self.params = params or {}
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    # ----------------------------------------------------------------------
    # Core abstract interface
    # ----------------------------------------------------------------------
    @abstractmethod
    def compute(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute kernel matrix between X and Y."""
        pass

    # ----------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------
    def validate_inputs(self, X: np.ndarray, Y: Optional[np.ndarray]) -> None:
        """Ensure X and Y are valid 2D numpy arrays."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if Y is not None and not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (samples × features).")
        if Y is not None and Y.ndim != 2:
            raise ValueError("Y must be a 2D array (samples × features).")

    # ----------------------------------------------------------------------
    # Parameter and config handling
    # ----------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """Return kernel configuration as a serializable dictionary."""
        return {"name": self.name, "params": self.params, "seed": self.seed}

    def set_params(self, **kwargs: Any) -> None:
        """Update kernel parameters dynamically."""
        self.params.update(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ----------------------------------------------------------------------
    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({param_str})"

