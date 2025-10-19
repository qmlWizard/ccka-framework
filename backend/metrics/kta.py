import numpy as np


def center_kernel(K: np.ndarray) -> np.ndarray:
    """
    Center a kernel matrix in feature space using double centering.

    Formula:
        Kc = K - 1*K - K*1 + 1*K*1
    where 1 is a matrix of ones / n.
    """
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n


def kta(Kx: np.ndarray, Ky: np.ndarray, centered: bool = True) -> float:
    """
    Compute Kernel Target Alignment (KTA) between two kernel matrices.

    KTA measures how similar two kernels are in feature space,
    normalized by their Frobenius norms.

    Args:
        Kx (np.ndarray): First kernel matrix (n × n).
        Ky (np.ndarray): Second kernel matrix (n × n).
        centered (bool, optional): If True, apply kernel centering before computing alignment.
                                   Defaults to True.

    Returns:
        float: Alignment score in the range [-1, 1].

    Raises:
        ValueError: If kernel matrix shapes do not match.

    Formula:
        KTA(Kx, Ky) = ⟨Kx, Ky⟩ / (‖Kx‖_F * ‖Ky‖_F)
    """
    if Kx.shape != Ky.shape:
        raise ValueError(f"Kernel shapes must match: got {Kx.shape} vs {Ky.shape}")

    if centered:
        Kx = center_kernel(Kx)
        Ky = center_kernel(Ky)

    numerator = np.sum(Kx * Ky)
    denominator = np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky))

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)
