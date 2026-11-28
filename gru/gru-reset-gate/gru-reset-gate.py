import torch

def reset_gate(h_prev: torch.Tensor, x_t: torch.Tensor,
               W_r: torch.Tensor, b_r: torch.Tensor) -> torch.Tensor:
    """
    Returns the reset-gate values with the shape and dtype of h_prev.
    """
    joined = torch.cat([h_prev, x_t], dim=-1)
    return torch.sigmoid(joined @ W_r.T + b_r)