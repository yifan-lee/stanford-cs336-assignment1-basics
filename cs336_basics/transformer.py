import torch

import torch.nn as nn
import numpy as np

from einops import rearrange, einsum



class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(
            torch.empty(out_features, in_features, device=self.device, dtype=self.dtype)
        )
        std = np.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(
            self.weights,
            mean = 0,
            std=std,
            a=-3*std,
            b=3*std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply the linear transformation to the input
        '''
        output = einsum(
            x, self.weights,
            "... in_features, out_features in_features -> ... out_features"
        )
        return output

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=self.device, dtype=self.dtype)
        )
        nn.init.trunc_normal_(
            self.weights,
            mean = 0,
            std=1,
            a=-3,
            b=3
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        output = self.weights[token_ids]
        return output
