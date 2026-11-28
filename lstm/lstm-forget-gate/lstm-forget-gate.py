import torch

def forget_gate(h_prev: torch.Tensor, x_t: torch.Tensor,
                W_f: torch.Tensor, b_f: torch.Tensor) -> torch.Tensor:
    """
    Returns the forget-gate values with the shape and dtype of h_prev.
    """
    joined = torch.concatenate([h_prev, x_t], axis=-1)
    return torch.sigmoid(joined @ W_f.T + b_f)