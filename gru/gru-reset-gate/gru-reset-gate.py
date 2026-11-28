import torch

def reset_gate(h_prev: torch.Tensor, x_t: torch.Tensor,
               W_r: torch.Tensor, b_r: torch.Tensor) -> torch.Tensor:
    """
    Returns the reset-gate values with the shape and dtype of h_prev.
    """
    joined = torch.concatenate([h_prev, x_t], axis=-1)
    z = W_r @ joined.T + b_r
    return torch.sigmoid(z.T)