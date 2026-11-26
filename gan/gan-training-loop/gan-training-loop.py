import numpy as np

def train_discriminator_step(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    D_W: np.ndarray,
    learning_rate: float,
) -> dict:
    """
    Returns updated discriminator weights and the pre-update loss.
    """
    p_real = 1 / (1 + np.exp(-(real_data @ D_W)))
    p_fake = 1 / (1 + np.exp(-(fake_data @ D_W)))

    new_discriminator_weights = (real_data.T @ (p_real - 1) + fake_data.T @ p_fake) / real_data.shape[0]

    p_real = np.clip(p_real, 1e-8, 1.0 - 1e-8)
    p_fake = np.clip(p_fake, 1e-8, 1.0 - 1e-8)
    
    discriminator_loss = -np.mean(np.log(p_real) + np.log(1 - p_fake))

    return {"new_discriminator_weights": D_W - learning_rate * new_discriminator_weights,"discriminator_loss":discriminator_loss}