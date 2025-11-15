
from __future__ import annotations
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Tuple, Optional, Deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

Transition = namedtuple('Transition', ('state','noise', 'action', 'reward', 'next_state', 'next_noise', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        states = torch.tensor(np.array([b.state for b in batch]), dtype=torch.float32, device=self.device)
        # times = torch.tensor(np.array([b.time for b in batch]), dtype=torch.float32, device=self.device)
        noises = torch.tensor(np.stack([np.asarray(b.noise, dtype=np.float32) for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([b.action for b in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([b.reward for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        next_noises = torch.tensor(np.stack([np.asarray(b.next_noise, dtype=np.float32) for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([b.done for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
        return states,  noises, actions, rewards, next_states, next_noises, dones

    def __len__(self):
        return len(self.buffer)

def set_seed(env, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # env.seed(seed)
    env.action_space.seed(seed)
    try:
        env.observation_space.seed(seed)
    except Exception:
        pass

def adjust_gamma(original_gamma: float, step_ratio: float) -> float:
    """
    Adjusts the discount factor gamma when the timestep changes.

    Parameters:
    ----------
    original_gamma : float
        The original discount factor (e.g., 0.99)
    step_ratio : float
        How many new steps correspond to one old step.
        For example:
            - If the new horizon has twice as many steps per unit time, step_ratio = 2
            - If you now take half as many steps per unit time, step_ratio = 0.5

    Returns:
    -------
    new_gamma : float
        The adjusted gamma that keeps the same real-time discount rate.
    """
    new_gamma = original_gamma ** (1.0 / step_ratio)
    return new_gamma    
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class MLP(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()
        layers = []
        last = inp_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)
    
    
class VelNet(nn.Module):
    def __init__(self, state_dim=2,action_dim=2, hidden=256, with_std=True):
        super().__init__()
        self.fc1 = nn.Linear(state_dim+action_dim+1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)
        self.head_std = nn.Linear(hidden, action_dim)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.head_std.bias, -3e-3, 3e-3)
        self.with_std = with_std
    def forward(self, x, t):  # x: [B,s_d+a_d], t: [B,1]
        assert x.device == t.device, f"Device mismatch: x on {x.device}, t on {t.device}"


        h = torch.cat([x, t], dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        mean = self.fc3(h)
        std = torch.exp(self.head_std(h)) if self.with_std else None

        return mean, std
    
class QNetwork_FSAC(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(256,256), with_time = False):
        super().__init__()
        self.with_time = with_time
        if with_time:
            self.q = MLP(state_dim + action_dim + action_dim+ 1, 1, hidden_sizes)
        else:
          self.q = MLP(state_dim + action_dim + action_dim , 1, hidden_sizes)

    def forward(self, state ,action,t = None):
        if self.with_time:
          assert t is not None, "t must be provided if with_time is True"
          x = torch.cat([state, action,t], dim=-1)
        else:
          x = torch.cat([state, action], dim=-1)
        return self.q(x)