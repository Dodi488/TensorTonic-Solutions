import numpy as np

def vae_loss(x: np.ndarray, reconstruction: np.ndarray,
             mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns total_loss, reconstruction_loss, and kl_loss as Python floats.
    """
    B = x.shape[0]
    
    L_reco = np.sum(np.sum(np.square(x - reconstruction), axis=1), axis=0)
    L_reco /= B

    L_KL = -np.sum(np.sum(1 + log_var - np.square(mu) - np.exp(log_var), axis=1), axis=0)
    L_KL /= 2 * B

    L = L_reco + L_KL

    return  {"total_loss":L,"reconstruction_loss":L_reco,"kl_loss":L_KL}