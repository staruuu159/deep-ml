import torch

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    d_k = Q.shape[-1]

    scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
    attention = torch.softmax(scores, dim=-1)
    output = attention @ V
    return output