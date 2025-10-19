import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


def full_matrix(
    kernel_fn: Callable[[np.ndarray, np.ndarray], float],
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    Compute the full kernel matrix K(X, Y) in parallel using multithreading.

    Automatically allocates ~80% of available CPU cores for efficient computation,
    balancing throughput and system responsiveness.

    Args:
        kernel_fn (Callable[[np.ndarray, np.ndarray], float]):
            Function that computes kernel similarity between two feature vectors.
            Example: lambda x, y: np.exp(-gamma * np.linalg.norm(x - y) ** 2)
        X (np.ndarray):
            Data matrix of shape (n_samples_X, n_features).
        Y (np.ndarray):
            Data matrix of shape (n_samples_Y, n_features).

    Returns:
        np.ndarray:
            Kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError("Inputs X and Y must be NumPy arrays.")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays (samples × features).")

    # ---- Determine thread usage based on system cores ---------------------
    n_cores = multiprocessing.cpu_count()
    n_threads = max(1, int(n_cores * 0.8))
    n_x, n_y = X.shape[0], Y.shape[0]
    K = np.zeros((n_x, n_y), dtype=np.float64)

    # ---- Define work per thread ------------------------------------------
    chunk_size = max(1, n_x // n_threads)

    def _compute_chunk(start_idx: int, end_idx: int) -> tuple[int, int, np.ndarray]:
        """Compute a slice of the kernel matrix."""
        sub_K = np.zeros((end_idx - start_idx, n_y), dtype=np.float64)
        for i, x in enumerate(X[start_idx:end_idx]):
            sub_K[i, :] = [kernel_fn(x, y) for y in Y]
        return start_idx, end_idx, sub_K

    # ---- Parallel execution ----------------------------------------------
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            executor.submit(_compute_chunk, start, min(start + chunk_size, n_x))
            for start in range(0, n_x, chunk_size)
        ]
        for f in futures:
            start_idx, end_idx, sub_K = f.result()
            K[start_idx:end_idx, :] = sub_K

    return K
