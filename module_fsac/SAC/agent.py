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

from .utily import init_weights, ReplayBuffer, QNetwork
from .policy import GaussianPolicy



@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005                # target smoothing coefficient (polyak)
    alpha: Optional[float] = None     # if None -> automatic entropy tuning
    lr: float = 3e-4
    batch_size: int = 256
    hidden_sizes: Tuple[int, ...] = (256, 256)
    replay_size: int = 1_000_000
    start_steps: int = 10000         # steps using random policy before using learned policy
    update_after: int = 1000         # start gradient updates after this many steps
    update_every: int = 50           # number of env steps per update (update_every gradient steps)
    policy_delay: int = 1            # how often to update policy (policy_delay not strictly necessary for SAC)
    target_update_interval: int = 1

class SACAgent:
    def __init__(self, obs_dim, action_dim, action_low, action_high, device: torch.device, config: SACConfig):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.config = config

        # networks
        self.policy = GaussianPolicy(obs_dim, action_dim, hidden_sizes=config.hidden_sizes).to(device)
        self.q1 = QNetwork(obs_dim, action_dim, hidden_sizes=config.hidden_sizes).to(device)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_sizes=config.hidden_sizes).to(device)
        # target networks for critics
        self.q1_target = QNetwork(obs_dim, action_dim, hidden_sizes=config.hidden_sizes).to(device)
        self.q2_target = QNetwork(obs_dim, action_dim, hidden_sizes=config.hidden_sizes).to(device)
        # copy weights
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=config.lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=config.lr)

        # automatic entropy tuning
        if config.alpha is None:
            # target entropy heuristic: -|A|
            self.target_entropy = -float(action_dim)
            # log alpha parameter
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=config.lr)
        else:
            self.alpha = config.alpha
            self.log_alpha = None
            self.alpha_optim = None
            self.target_entropy = None

    @property
    def alpha(self):
        if self.log_alpha is None:
            return self._alpha_fixed
        else:
            return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value):
        self._alpha_fixed = value

    def select_action(self, state: np.ndarray, deterministic: bool=False) -> np.ndarray:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.policy(state_t, deterministic=deterministic)
        action = action.cpu().numpy()[0]
        return action

    def update(self, replay: ReplayBuffer):
        cfg = self.config
        if len(replay) < cfg.batch_size:
            return {}

        states, actions, rewards, next_states, dones = replay.sample(cfg.batch_size)

        # actions from replay are in env scale; convert to unscaled tanh space [-1,1] for Q input consistency
        actions_unscaled = torch.tensor(actions.cpu().numpy(), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # sample next action from policy, compute log prob
            next_action, next_log_prob, _ = self.policy(next_states)
            # scale next_action to env action space for Q calculation (critics expect env scaled? We use unscaled consistently)
            # We will keep Q networks consistent with unscaled [-1,1] action representation
            # compute target Q value using targets
            q1_next_target = self.q1_target(next_states, next_action)
            q2_next_target = self.q2_target(next_states, next_action)
            q_next_target = torch.min(q1_next_target, q2_next_target) - (self.alpha.detach() * next_log_prob)
            target_q = rewards + (1.0 - dones) * cfg.gamma * q_next_target

        # Critic losses
        q1_pred = self.q1(states, actions_unscaled)
        q2_pred = self.q2(states, actions_unscaled)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optim.zero_grad(); q1_loss.backward(); self.q1_optim.step()
        self.q2_optim.zero_grad(); q2_loss.backward(); self.q2_optim.step()

        # Policy loss (sampled from states)
        new_action, log_prob, _ = self.policy(states)
        # note: q networks expect unscaled actions in [-1,1]
        q1_new = self.q1(states, new_action)
        q2_new = self.q2(states, new_action)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Entropy (alpha) tuning
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.log_alpha is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        # Soft update of target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(cfg.tau * param.data + (1.0 - cfg.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(cfg.tau * param.data + (1.0 - cfg.tau) * target_param.data)

        info = {
            'loss/q1_loss': q1_loss.item(),
            'loss/q2_loss': q2_loss.item(),
            'q_detail/q1_mean': q1_pred.mean().item(),
            'q_detail/q2_mean': q2_pred.mean().item(),
            'q_detail/q1_std': q1_pred.std().item(),
            'q_detail/q2_std': q2_pred.std().item(),
            'q_detail/q1_min': q1_pred.min().item(),
            'q_detail/q1_max': q1_pred.max().item(),
            'q_detail/q2_min': q2_pred.min().item(),
            'q_detail/q2_max': q2_pred.max().item(),
            'loss/policy_loss': policy_loss.item(),
            'actor/policy_output': new_action.mean().item(),
            'loss/alpha_loss': alpha_loss.item() if isinstance(alpha_loss, torch.Tensor) else float(alpha_loss),
            'actor/alpha': self.alpha.item() if self.log_alpha is not None else float(self.alpha)
        }
        if new_action.ndim == 2:  # [batch_size, action_dim]
          for i in range(new_action.shape[1]):
              info.update({
                  f'actor/action_mean_{i}': new_action[:, i].mean().item(),
                  f'actor/action_std_{i}': new_action[:, i].std().item(),
                  f'actor/action_min_{i}': new_action[:, i].min().item(),
                  f'actor/action_max_{i}': new_action[:, i].max().item(),
              })
        return info

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, 'policy.pth'))
        torch.save(self.q1.state_dict(), os.path.join(path, 'q1.pth'))
        torch.save(self.q2.state_dict(), os.path.join(path, 'q2.pth'))
        if self.log_alpha is not None:
            torch.save(self.log_alpha.detach().cpu(), os.path.join(path, 'log_alpha.pth'))

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth'), map_location=self.device))
        self.q1.load_state_dict(torch.load(os.path.join(path, 'q1.pth'), map_location=self.device))
        self.q2.load_state_dict(torch.load(os.path.join(path, 'q2.pth'), map_location=self.device))
        if self.log_alpha is not None and os.path.exists(os.path.join(path, 'log_alpha.pth')):
            self.log_alpha.data = torch.load(os.path.join(path, 'log_alpha.pth'), map_location=self.device)
