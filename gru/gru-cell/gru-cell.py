import torch

def gru_cell(x_t: torch.Tensor, h_prev: torch.Tensor,
             W_r: torch.Tensor, W_z: torch.Tensor, W_h: torch.Tensor,
             b_r: torch.Tensor, b_z: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    """
    Returns the next hidden state with the shape of h_prev and the dtype of x_t.
    """
    joined = torch.cat([h_prev, x_t], dim=-1)
    r_t = torch.sigmoid(joined @ W_r.T + b_r)
    z_t = torch.sigmoid(joined @ W_z.T + b_z)
    new_joined = torch.cat([(r_t * h_prev), x_t], dim=-1)
    h_tilde = torch.tanh(new_joined @ W_h.T + b_h)
    return z_t * h_prev + (1 - z_t) * h_tilde