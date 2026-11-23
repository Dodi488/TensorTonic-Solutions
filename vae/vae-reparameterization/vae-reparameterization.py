import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """
    Returns the float64 latent sample with the same shape as mu.
    """
    sigma = np.exp(log_var / 2)
    return mu + sigma * epsilon