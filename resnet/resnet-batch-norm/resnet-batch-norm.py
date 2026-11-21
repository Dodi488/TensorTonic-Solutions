import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns the normalized residual-block result and selected mode in a dictionary.
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    gamma1 = np.array(gamma1)
    beta1 = np.array(beta1)
    gamma2 = np.array(gamma2)
    beta2 = np.array(beta2)
    if mode == "post":
        # Conv
        z = x @ W1
    elif mode == "pre":
        z = x

    # BN
    x_hat = (z - np.mean(z, axis=0, keepdims=True)) / np.sqrt(np.var(z, axis=0, keepdims=True) + 1e-5)
    BN = gamma1 * x_hat + beta1

    # ReLu
    RELU = np.maximum(0, BN)

    if mode == "post":
        # Conv
        z = RELU @ W2
    elif mode == "pre":
        z = RELU @ W1

    # BN
    x_hat = (z - np.mean(z, axis=0, keepdims=True)) / np.sqrt(np.var(z, axis=0, keepdims=True) + 1e-5)
    BN = gamma2 * x_hat + beta2

    if mode == "post":
        # Add
        skip = x + BN

        # ReLU
        final = np.maximum(0, skip)
    elif mode == "pre":
        # ReLU
        RELU = np.maximum(0, BN)

        # Conv
        conv = RELU @ W2

        # Add
        final = x + conv

    return {"output": final, "mode": mode}