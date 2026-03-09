import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_model = Q.shape[2]
    qk = (Q @ K.transpose(-2, -1)) / math.sqrt(d_model)
    qk_softmax = F.softmax(qk, dim=-1)
    attention = qk_softmax @ V

    return attention