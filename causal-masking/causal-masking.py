import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Write code here
    scores = np.array(scores)
    
    attention_mask_lower = np.tril(scores)
    attention_mask_upper = np.triu(np.ones_like(scores), k=1) * mask_value

    return attention_mask_lower + attention_mask_upper

    