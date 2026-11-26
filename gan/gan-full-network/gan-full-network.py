import numpy as np

def gan_forward(
    z: np.ndarray,
    real_data: np.ndarray,
    G_W: np.ndarray,
    G_b: np.ndarray,
    D_W: np.ndarray,
) -> dict:
    """
    Returns generated samples, probabilities, and both GAN losses.
    """
    fake_data = np.tanh(z @ G_W + G_b)

    p_real = 1 / (1 + np.exp(-(real_data @ D_W)))
    p_fake = 1 / (1 + np.exp(-(fake_data @ D_W)))

    p_real = np.clip(p_real, 1e-8, 1.0 - 1e-8)
    p_fake = np.clip(p_fake, 1e-8, 1.0 - 1e-8)
    
    discriminator_loss = -np.mean(np.log(p_real) + np.log(1 - p_fake))

    generator_loss = -np.mean(np.log(p_fake))
    
    return {"generated_samples":fake_data,"real_probabilities":p_real,"fake_probabilities":p_fake,"discriminator_loss":discriminator_loss,"generator_loss":generator_loss}