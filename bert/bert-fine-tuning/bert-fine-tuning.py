import numpy as np

def bert_fine_tuning_step(hidden_states: np.ndarray, labels: np.ndarray,
                          classifier_W: np.ndarray, classifier_b: np.ndarray,
                          learning_rate: float) -> dict:
    """
    Returns updated classifier parameters and the pre-update loss.
    """
    H = hidden_states[:, 0, :]
    z = H @ classifier_W + classifier_b
    exponentials = np.exp(z - np.max(z, axis=1, keepdims=True))
    p = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    batch_size = labels.shape[0]
    l = -np.mean(np.log(p[np.arange(batch_size), labels]))
    grad = p.copy()
    grad[np.arange(batch_size), labels] -= 1.0
    grad /= batch_size
    grad_W = H.T @ grad
    grad_b = np.sum(grad, axis=0)

    new_classifier_W = classifier_W - learning_rate * grad_W
    new_classifier_b = classifier_b - learning_rate * grad_b
    
    return {"new_classifier_W":new_classifier_W,"new_classifier_b":new_classifier_b,"loss":l} 