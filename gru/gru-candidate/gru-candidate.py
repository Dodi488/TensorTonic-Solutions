import torch

def candidate_hidden(r_t: torch.Tensor, h_prev: torch.Tensor, x_t: torch.Tensor,
                     W_h: torch.Tensor, b_h: torch.Tensor) -> torch.Tensor:
    """
    Returns the candidate hidden state with the shape and dtype of h_prev.
    """
    joined = torch.cat([(r_t * h_prev), x_t], dim=-1)
    return torch.tanh(joined @ W_h.T + b_h)