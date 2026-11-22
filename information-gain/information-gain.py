import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    # Write code here
    y = np.array(y)
    split_mask = np.array(split_mask)
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    p = (counts / n)
    H = -np.sum(p * np.log2(p))

    y_left = y[split_mask]
    y_right = y[~split_mask]

    classes_left, counts_left = np.unique(y_left, return_counts=True)
    p_left = (counts_left / len(y_left))

    classes_right, counts_right = np.unique(y_right, return_counts=True)
    p_right = (counts_right / len(y_right))

    w_left = len(y_left) / n
    w_right = len(y_right) / n

    H_left = -np.sum(p_left * np.log2(p_left))
    H_right = -np.sum(p_right * np.log2(p_right))

    H_children = w_left * H_left + w_right * H_right
    return H - H_children