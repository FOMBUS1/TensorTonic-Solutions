import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = np.zeros([seq_len, d_model])
    for pos in range(seq_len):
      for i in range(d_model):
        denom = base ** ((2 * (i//2)) / d_model)
        if i % 2 == 0:
          pe[pos, i] = np.sin(pos / denom)
        else:
          pe[pos, i] = np.cos(pos/denom)
    return pe