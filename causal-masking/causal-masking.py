import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    # Write code here
    scores = np.array(scores)
    mask = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    return np.where(mask, mask_value, scores)