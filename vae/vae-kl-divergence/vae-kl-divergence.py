import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Returns the batch-mean KL divergence as a Python float.
    """
    logits = 1 + log_var - np.square(mu) - np.exp(log_var)
    sum = np.sum(np.sum(logits, axis=1), axis=0)
    return -sum / (2 * mu.shape[0])