import numpy as np

def apply_mlm_mask(token_ids: np.ndarray, mask_positions: np.ndarray,
                   replace_probs: np.ndarray, random_tokens: np.ndarray,
                   mask_token_id: int = 103) -> dict:
    """
    Returns masked_ids and labels as int64 arrays in a dictionary.
    """
    masked_id = token_ids.copy()
    labels = np.full_like(token_ids, -100)
    for i in range(len(masked_id)):
        for j in range(len(masked_id[i])):
            if mask_positions[i][j]:
                labels[i][j] = masked_id[i][j]
                if replace_probs[i][j] < 0.8:
                    masked_id[i][j] = mask_token_id
                elif 0.8 <= replace_probs[i][j] < 0.9:
                    masked_id[i][j] = random_tokens[i][j]
    return {"masked_ids":masked_id,"labels":labels}