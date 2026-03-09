import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    u = np.mean(x, axis=-1, keepdims=True)
    d = np.var(x, axis=-1, keepdims=True)
    return (gamma * (x - u) / (np.sqrt(d + eps))) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """

    # Linear projections
    Q = Q @ W_q
    K = K @ W_k
    V = V @ W_v

    batch, seq, d_model = Q.shape
    dk = d_model // num_heads

    # Split heads
    Q = Q.reshape(batch, seq, num_heads, dk).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq, num_heads, dk).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq, num_heads, dk).transpose(0, 2, 1, 3)

    # Attention scores
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dk)

    # Attention weights
    attention = softmax(scores)

    # Apply attention to values
    context = attention @ V

    # Concatenate heads
    context = context.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)

    # Final projection
    output = context @ W_o

    return output

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    hidden = x@W1 + b1
    relu = np.maximum(0, hidden)
    output = relu@W2 + b2
    return output

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    mla = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads=num_heads), gamma1, beta1)
    output = layer_norm(mla + feed_forward(mla, W1, b1, W2, b2), gamma2, beta2)

    return output