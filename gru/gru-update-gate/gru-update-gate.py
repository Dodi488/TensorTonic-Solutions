import torch

def update_gate(h_prev: torch.Tensor, x_t: torch.Tensor,
                W_z: torch.Tensor, b_z: torch.Tensor) -> torch.Tensor:
    """
    Returns the update-gate values with the shape and dtype of h_prev.
    """
    joined = torch.cat([h_prev, x_t], dim=-1)
    return torch.sigmoid(joined @ W_z.T + b_z)