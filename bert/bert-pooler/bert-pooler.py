import numpy as np

def bert_pooler(hidden_states: np.ndarray, W_pool: np.ndarray,
                b_pool: np.ndarray) -> np.ndarray:
    """
    Returns the float64 pooled states with shape (B, D).
    """
    H = hidden_states[:, 0, :]
    return np.tanh(H @ W_pool + b_pool)