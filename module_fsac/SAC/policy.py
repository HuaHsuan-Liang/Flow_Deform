from __future__ import annotations
import argparse
import math
import os
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Tuple, Optional, Deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from .utily import init_weights

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
    
    
class GaussianPolicy(nn.Module):
    """
    Gaussian policy with reparameterization and tanh squashing.
    Returns action, log_prob, and mean (pre-tanh mean).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(256,256), log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        last = state_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.net = nn.Sequential(*layers)
        self.mean_linear = nn.Linear(last, action_dim)
        self.log_std_linear = nn.Linear(last, action_dim)
        self.net.apply(init_weights)
        nn.init.uniform_(self.mean_linear.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_linear.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, deterministic: bool=False):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            # Use a small constant for log_prob when deterministic
            log_prob = None
            return action, log_prob, mean

        # reparameterization trick
        normal = Normal(mean, std)
        z = normal.rsample()  # sample with reparam
        action = torch.tanh(z)

        # Log prob correction for tanh squashing
        # log_prob(z) - sum(log(1 - tanh(z)^2) + eps)
        log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)
        # Numerically stable correction
        eps = 1e-6
        log_prob -= torch.log(1 - action.pow(2) + eps).sum(dim=-1, keepdim=True)
        return action, log_prob, mean