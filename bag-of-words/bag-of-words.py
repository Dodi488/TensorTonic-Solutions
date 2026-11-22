import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    # Write code here
    tokens = np.array(tokens)
    vocab = np.array(vocab)
    return (vocab[:, None] == tokens).sum(axis=1)