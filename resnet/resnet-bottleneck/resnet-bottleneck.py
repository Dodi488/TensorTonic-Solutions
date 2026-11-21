import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns the bottleneck residual-block output as a nested list.
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    Ws = np.array(Ws)

    return np.maximum(0, np.maximum(0, np.maximum(0, x @ W1) @ W2) @ W3 + x @ Ws)