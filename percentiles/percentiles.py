import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns a NumPy array of percentiles.
    """
    # Write code here
    x = np.sort(np.array(x))
    q = np.array(q)

    r = q / 100 * (x.size - 1)
    l = np.floor(r).astype(int)
    u = np.ceil(r).astype(int)
    w = r - l

    return x[l] * (1 - w) + x[u] * w