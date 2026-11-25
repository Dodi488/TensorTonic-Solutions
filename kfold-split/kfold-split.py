import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    # Write code here
    rng = np.random.default_rng(seed)
    data = np.arange(N)
    if shuffle:
        data = rng.permutation(data)
    fold_size = np.array_split(data, k)
    scores = []
    
    for i in range(k):
        val_idx = fold_size[i]
        train_idx = np.concatenate([fold_size[j] for j in range(k) if j != i])

        scores.append({"train_idx": train_idx, "val_idx": val_idx})

    return scores