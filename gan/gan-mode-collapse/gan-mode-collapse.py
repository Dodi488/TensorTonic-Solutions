import numpy as np

def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Returns diversity_score and is_collapsed in a dictionary.
    """
    sj = np.std(generated_samples, axis=0)
    s = np.mean(sj, axis=0)
    return {"diversity_score":s,"is_collapsed":bool(s < threshold)}