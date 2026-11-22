import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    X = np.array(X_train, dtype=float)
    q = np.array(X_test, dtype=float)
    ntrain = len(X_train)
    if X.ndim == 1:
        X = X[:, None]
        if q.ndim == 1:
            q = q[:, None]
            
    dists = np.sqrt(np.sum((q[:, None, :] - X[None, :, :]) ** 2, axis=2))
    nearest = np.argsort(dists, axis=1, kind="stable")[:, :k]

    if k > ntrain:
        padding = np.full((q.shape[0], k - ntrain), -1)
        nearest = np.concatenate([nearest, padding], axis=1)

    return nearest