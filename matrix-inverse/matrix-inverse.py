import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    # Write code here
    A = np.array(A, dtype=float)
    n = A.shape[0]
    det = np.linalg.det(A)
    if np.isclose(det, 0):
        return None

    if n == 1:
        return np.array([[1.0 / A[0, 0]]])

    cof = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            cof[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)

    return cof.T / det