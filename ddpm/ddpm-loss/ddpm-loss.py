import numpy as np

def compute_ddpm_loss(epsilon: list, epsilon_pred: list) -> float:
    """
    Returns the mean DDPM noise-prediction loss.
    """
    epsilon = np.array(epsilon)
    epsilon_pred = np.array(epsilon_pred)
    return np.mean(np.square(epsilon - epsilon_pred))