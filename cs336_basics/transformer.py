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

class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weights = nn.Parameter(
            torch.ones(d_model, device=self.device, dtype=self.dtype)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms+self.eps)
        result = x/rms * self.weights
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype


        self.w1_weight = Linear(self.d_model,self.d_ff,self.device,self.dtype)
        self.w2_weight = Linear(self.d_ff,self.d_model,self.device,self.dtype)
        self.w3_weight = Linear(self.d_model,self.d_ff,self.device,self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def SiLU(x: torch.Tensor) -> torch.Tensor:
            return x*torch.sigmoid(x)
        
        x1 = self.w1_weight(x)
        x1_silu = SiLU(x1)
        x3 = self.w3_weight(x)
        x1_silu_x3 = x1_silu*x3
        result = self.w2_weight(x1_silu_x3)
        return result

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, 
        theta: float, 
        d_k: int, 
        max_seq_len: int,
        device: torch.device | None = None, 
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        dim_index = torch.arange(self.d_k // 2, device=self.device, dtype=torch.float32)
        position_index = torch.arange(self.max_seq_len, device=self.device, dtype=torch.float32)
        theta_inv_index = self.theta**(-2*dim_index/d_k)
        theta_ik = einsum(
            position_index, theta_inv_index,
            "s, d -> s d"
        )


        sin = torch.sin(theta_ik)
        cos = torch.cos(theta_ik)
        
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
    
    def forward(
        self, x: torch.Tensor,
        toke_position: torch.Tensor,
    ) -> torch.Tensor:
        x_even = x[...,::2]
        x_odd = x[...,1::2]

        sin_expend = self.sin[toke_position]
        cos_expend = self.cos[toke_position]

        x_even_new = x_even*cos_expend-x_odd*sin_expend
        x_odd_new = x_even*sin_expend+x_odd*cos_expend

        x_rope = rearrange(
            torch.stack([x_even_new,x_odd_new], dim=-1),
            '... seq_len d_k two -> ... seq_len (d_k two)',
        )
        return x_rope

def softmax(
    x: torch.Tensor,
    dimension: int,    
):
    max_values, _ = torch.max(x, dim=dimension, keepdim=True)
    x_exp = torch.exp(x-max_values)
    x_rxp_sum = torch.sum(x_exp, dim=dimension, keepdim=True)
    return x_exp/x_rxp_sum