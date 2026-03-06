import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not(seqs):
        return (0, 0)
    L = max_len if max_len is not None else max(len(seq) for seq in seqs)
    padded_seqs = []
    for seq in seqs:
        seq += [pad_value] * (L - len(seq))
        seq = seq[:L]
        padded_seqs.append(seq)

    return padded_seqs