import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.array(x)

    if len(x.shape) == 4:
        N, C, H, W = x.shape
        axis = (0, 2, 3)
        beta = np.array(beta).reshape(1, C, 1, 1)
        gamma = np.array(gamma).reshape(1, C, 1, 1)
    else:
        axis = 0
    
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    std = np.sqrt(var + eps)
    x_hat = (x - mean) / std
    return gamma * x_hat + beta