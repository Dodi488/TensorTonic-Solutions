import torch

def hidden_state_update(z_t: torch.Tensor, h_prev: torch.Tensor,
                        h_tilde: torch.Tensor) -> torch.Tensor:
    """
    Returns the updated hidden state with the shape and dtype of the inputs.
    """
    return z_t * h_prev + (1 - z_t) * h_tilde