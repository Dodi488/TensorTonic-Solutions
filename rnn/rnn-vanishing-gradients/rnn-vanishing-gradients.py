import torch

def compute_gradient_norm_decay(T: int, W_hh: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor with shape (T,) and the dtype of W_hh holding the spectral norm of W_hh raised to the powers 0 through T - 1.
    """
    gk = torch.linalg.matrix_norm(W_hh, ord=2)
    k = torch.arange(T, dtype=W_hh.dtype)
    return gk ** k