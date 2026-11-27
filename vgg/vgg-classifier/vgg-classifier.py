import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray,
                   W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C_classes).
    """
    x = features.reshape(features.shape[0], -1)
    h1 = np.maximum(0, x @ W1 + b1)
    h2 = np.maximum(0, h1 @ W2 + b2)
    return h2 @ W3 + b3