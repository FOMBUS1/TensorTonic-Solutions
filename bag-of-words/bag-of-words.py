import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    vector = np.zeros(len(vocab), dtype=np.int64)

    for idx, token in enumerate(vocab):
        vector[idx] = tokens.count(token)

    return vector
        