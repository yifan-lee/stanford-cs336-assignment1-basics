import torch

import torch.nn as nn
import numpy as np

from einops import rearrange, einsum


def cross_entropy(
    prediction: torch.Tensor,
    target: torch.Tensor
):
    max_values, _ = torch.max(prediction, dim = -1, keepdim=True)
    prediction_scaled = prediction - max_values
    prediction_scaled_exp = torch.exp(prediction_scaled)
    prediction_scaled_exp_sum = torch.sum(prediction_scaled_exp, dim = -1)
    prediction_scaled_exp_sum_log = torch.log(prediction_scaled_exp_sum)
    target_expend = rearrange(
        target,
        "... -> ... 1"
    )
    target_logits = torch.gather(prediction_scaled, -1, target_expend)
    target_logits = rearrange(
        target_logits,
        "... 1 -> ..."
    )
    result = - target_logits + prediction_scaled_exp_sum_log
    return torch.mean(result)