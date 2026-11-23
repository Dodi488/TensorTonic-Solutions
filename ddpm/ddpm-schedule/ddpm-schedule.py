import numpy as np

def linear_beta_schedule(T: int,
                         beta_1: float = 0.0001,
                         beta_T: float = 0.02) -> list[float]:
    """
    Returns T linearly spaced beta values.
    """
    if T == 1:
        return [round(beta_1, 6)]
    start = round(beta_1 + ((1 - 1) / (T - 1)) * (beta_T - beta_1), 6)
    end = round(beta_1 + ((T - 1) / (T - 1)) * (beta_T - beta_1), 6)
    final = np.round(np.linspace(start, end, T, dtype=float), 6)
    return final.tolist()