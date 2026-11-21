import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns the network logits as a nested list.
    """
    x = np.array(x)

    h = np.maximum(0, x @ conv1)
    h1 = np.maximum(0, np.maximum(0, h @ W1_b1) @ W2_b1 + h)

    a = np.maximum(0, h1 @ W1_b2)
    s = (a @ W2_b2) + (h1 @ Ws_b2)
    h2 = np.maximum(0, s)

    return h2 @ fc