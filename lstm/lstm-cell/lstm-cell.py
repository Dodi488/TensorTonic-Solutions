import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0+np.exp(-np.clip(x,-500,500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> dict:
    """
    Returns hidden_state and cell_state as float64 arrays.
    """
    joined = np.concatenate([h_prev, x_t], axis=-1)
    f_t = sigmoid(joined @ W_f.T + b_f)
    it = sigmoid(joined @ W_i.T + b_i)
    C_tilde = np.tanh(joined @ W_c.T + b_c)
    cell_state = f_t * C_prev + it * C_tilde
    ot = sigmoid(joined @ W_o.T + b_o)
    hidden_state = ot * np.tanh(cell_state)
    return {"hidden_state":hidden_state,"cell_state":cell_state}