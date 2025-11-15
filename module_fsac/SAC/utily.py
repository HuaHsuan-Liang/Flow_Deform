
from __future__ import annotations
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Tuple, Optional, Deque

import numpy as np
import torch
import torch.nn as nn
import math

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

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
        actions = torch.tensor(np.array([b.action for b in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([b.reward for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([b.done for b in batch]), dtype=torch.float32, device=self.device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

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
    
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(256,256)):
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1, hidden_sizes)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q(x)