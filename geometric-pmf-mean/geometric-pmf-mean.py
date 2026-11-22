import numpy as np

def geometric_pmf_mean(k: list, p: float) -> dict:
    """
    Returns a dictionary with pmf and mean.
    """
    # Write code here
    q = 1 - p
    new = [(q ** (k[i] - 1)) * p for i in range(len(k))]
    return {"pmf": np.array(new), "mean": (1 / p)}