import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g).copy()
    norm = np.linalg.norm(g)

    print(norm)
    
    if norm <= max_norm:
        return g
    return g if (max_norm <= 0 or norm <= 0) else g * (max_norm / norm)