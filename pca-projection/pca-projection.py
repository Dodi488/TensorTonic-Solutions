import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    # Write code here
    X = np.array(X)
    n, d = X.shape

    # Center the data
    X_c = X - np.mean(X, axis=0, keepdims=True)

    # Compute the covariance matrix
    C = (X_c.T @ X_c) / (n - 1)

    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]    
    W = eigenvectors[:, :k]

    # Project the data
    return X_c @ W