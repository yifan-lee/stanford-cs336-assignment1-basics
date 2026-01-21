import torch
import math

import torch.nn as nn
import numpy as np

from einops import rearrange, einsum
from typing import Optional
from collections.abc import Callable, Iterable


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

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params,
        lr=1e-3,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=0
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"] 
            weight_decay = group["weight_decay"] 
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g = p.grad.data


                m = state.get('m',torch.zeros_like(p.data))
                m = beta1*m+(1-beta1)*g
                state['m'] = m

                v = state.get('v',torch.zeros_like(p.data))
                v = beta2*v+(1-beta2)*g**2
                state['v'] = v

                t = state.get("t", 1)
                state['t'] = t+1

                lrt = lr*math.sqrt(1-beta2**t)/(1-beta1**t)

                p.data -= lrt*m/(torch.sqrt(v)+eps)
                if weight_decay != 0:
                    p.data = p.data-lr*weight_decay*p.data
        return loss

def learning_rate_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int
):
    if t<T_w:
        alpha_t = t*alpha_max/T_w
    elif t<= T_c:
        alpha_t = alpha_min + (alpha_max-alpha_min)*(1+math.cos((t-T_w)*math.pi/(T_c-T_w)))/2 
    else:
        alpha_t = alpha_min
    return alpha_t 


def gradient_clipping(
    parameters,
    max_norm: float,
    eps = 1e-6
):
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    total_norm = torch.norm(
        torch.stack([torch.norm(g, 2) for g in grads]), 
        2
    )

    clip_coef = max_norm / (total_norm + eps)

    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)