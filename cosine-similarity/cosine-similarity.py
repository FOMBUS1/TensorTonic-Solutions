import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.array(a)
    b = np.array(b)
    
    cosine = (a@b)/(np.linalg.norm(a) * np.linalg.norm(b)) if np.linalg.norm(a) * np.linalg.norm(b) != 0 else 0

    return cosine