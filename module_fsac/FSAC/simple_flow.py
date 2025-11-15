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


class FlowPolicy_simple(nn.Module):
    """
    Flow-based Gaussian policy using velocity integration.
    - Supports sampling from noise.
    - Supports flow-matching loss computation.
    """
    def __init__(self, state_dim: int, action_dim: int, steps=100, device="cpu"):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.steps = steps
        self.device = device
        self.vel_model = VelNet(state_dim, action_dim, with_std=False).to(device)
        self.dt = 1.0 / steps

    @torch.no_grad()
    def sample(self, state: torch.Tensor, z0: torch.Tensor = None):
        """
        Sample an action trajectory via flow integration.
        Returns:
          - action: final action after T steps
          - z0: initial noise
          - traj: list of intermediate states (for visualization)
        """
        state = state.to(self.device)
        if z0 is None:
            z0 = torch.randn(state.size(0), self.action_dim, device=self.device)

        z = z0.clone()
        traj = [z0.clone()]

        for i in range(self.steps):
            t = torch.full((state.size(0), 1), i / self.steps, device=self.device)
            state_t = torch.cat([state, z], dim=-1)
            # print(state_t.shape,t.shape)
            v, _ = self.vel_model(state_t, t)
            z = z + self.dt * v
            traj.append(z.clone())

        action = torch.tanh(z)
        return action, z0, traj

    def flow_loss(self, state, z0 ,action_target):
        """
        Compute flow-matching loss:
          - Randomly pick timesteps t ~ U(0,1)
          - Sample z_t as interpolation between noise and target
          - Predict velocity and match to ground truth
        """
        batch_size = state.size(0)
        losses = torch.zeros(batch_size, device=self.device)
        for i in range(20):
            t = torch.rand(batch_size, 1, device=self.device)
            # Interpolate between noise and target
            zt = (1 - t) * z0 + t * action_target
            # True velocity of the linear flow
            target_v = action_target - z0
            state_t = torch.cat([state, zt], dim=-1)
            # Model prediction
            v_pred, _ = self.vel_model(state_t, t)

            # MSE between predicted and true velocity
            loss = F.mse_loss(v_pred, target_v)
            losses += loss

        loss = losses.mean()
        return loss, {"flow_loss": loss.item()}