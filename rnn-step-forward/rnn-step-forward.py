import numpy as np

def rnn_step_forward(x_t: list, h_prev: list, Wx: list, Wh: list, b: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (H,).
    """
    # Write code here
    x_t = np.array(x_t)
    h_prev = np.array(h_prev)
    a = x_t @ Wx + h_prev @ Wh + b
    return np.tanh(a)