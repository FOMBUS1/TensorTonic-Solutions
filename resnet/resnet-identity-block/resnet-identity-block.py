import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    
    h = np.clip(x @ W1.T, a_min=0, a_max=None)
    y = np.clip(h @ W2.T + x, a_min=0, a_max=None)
    return y
