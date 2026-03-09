import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    Qi = Q@W_q
    Ki = K@W_k
    Vi = V@W_v

    dk = Q.shape[2] // num_heads

    Q = Q.reshape(Q.shape[0], num_heads, Q.shape[1], dk)
    K = K.reshape(K.shape[0], num_heads, K.shape[1], dk)
    V = V.reshape(V.shape[0], num_heads, V.shape[1], dk)

    scores = (Q@K.transpose(0, 1, 3, 2)) / np.sqrt(dk)
    attention = softmax(scores) @ V
    attention = attention.reshape(attention.shape[0], attention.shape[2], attention.shape[1]*attention.shape[3])
    
    return attention @ W_o