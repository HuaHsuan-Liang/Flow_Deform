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

from .utily import VelNet

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
    
    
class FlowPolicy(nn.Module):
    """
    Gaussian policy with reparameterization and tanh squashing.
    Returns action, log_prob, and mean (pre-tanh mean).
    """
    def __init__(self, state_dim: int, action_dim: int, log_std_min=-20, log_std_max=2,steps = 100,device="cpu"):
        super().__init__()
        self.log_std_min = log_std_min/steps
        self.log_std_max = log_std_max/steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.steps = steps
        self.device = device
        self.vel_model = VelNet(state_dim,action_dim).to(device)
        # self.log_std_vel= VelNet(state_dim,action_dim)
        self.dt = float(1)/steps

    def forward(self, state: torch.Tensor,t: torch.Tensor, deterministic: bool = False):
        # t = 0 ~ 1
        # print(state.shape,t.shape)
        state.to(self.device)
        t.to(self.device)
        mean,log_std = self.vel_model(state, t)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        # --- Deterministic case ---
        if deterministic:
            action_raw = mean
            action_tanh = torch.tanh(mean)
            return action_tanh, None, mean

        # --- Stochastic case ---
        normal = Normal(mean, std)
        z = normal.rsample()  # reparameterization trick
        # action_raw = z
        action_tanh = torch.tanh(z)

        # Per-sample condition: if t == 1 → tanh(z), else z
        # Compute log_prob
        log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)


        eps = 1e-6
        log_prob_correction = torch.log(1 - action_tanh.pow(2) + eps).sum(dim=-1, keepdim=True)
        log_prob -= log_prob_correction

        return action_tanh, log_prob, mean