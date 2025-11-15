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

from .utily import init_weights, ReplayBuffer, QNetwork_FSAC
from .policy import FlowPolicy



@dataclass
class FSACConfig:
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
    flow_steps: int = 20
    target_entropy: Optional[float] = None  # if None, use -|A|
    
class FSACAgent:
    def __init__(self, obs_dim, action_dim, action_low, action_high, device: torch.device,step:int ,config: FSACConfig):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.config = config
        self.steps = step
        self.dt = float(1)/self.steps
        # networks
        self.policy = FlowPolicy(obs_dim, action_dim,steps=step,device=device).to(device)
        self.q1 = QNetwork_FSAC(obs_dim, action_dim, hidden_sizes=config.hidden_sizes,with_time=True).to(device)
        self.q2 = QNetwork_FSAC(obs_dim, action_dim, hidden_sizes=config.hidden_sizes,with_time=True).to(device)
        # target networks for critics
        self.q1_target = QNetwork_FSAC(obs_dim, action_dim, hidden_sizes=config.hidden_sizes,with_time=True).to(device)
        self.q2_target = QNetwork_FSAC(obs_dim, action_dim, hidden_sizes=config.hidden_sizes, with_time=True).to(device)
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
            # self.target_entropy = -float(action_dim)
            self.target_entropy = config.target_entropy if config.target_entropy is not None else -float(action_dim)
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

    def select_action(self, state: np.ndarray,noise=None, deterministic: bool=False) -> np.ndarray:
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if noise is None:
          eps = torch.randn((state.shape[0],self.action_dim),device=self.device)
        else:
          eps = torch.tensor(noise, dtype=torch.float32, device=self.device).view(state.shape[0],self.action_dim)

        obs = torch.cat([state, eps], dim=-1)
        prev_action  = eps.clone()
        with torch.no_grad():
          for k in range(self.steps):
            t = torch.full((state.shape[0],1), (k)*self.dt,device=self.device)
            velocity, _, _ = self.policy(obs, t,deterministic=deterministic)
            # prev_action = self.extract_previous_action(state_t)
            # print("prev_action:",prev_action.shape)
            action = velocity*self.dt + prev_action
            # print("action:",action.shape)   
            action = torch.clamp(torch.tensor(action, dtype=torch.float32, device=self.device),-1,1)
            
            # state_t = torch.cat([state_t[:,:-self.action_dim], action], dim=-1)
            prev_action = action.clone()
            obs = torch.cat([state, prev_action], dim=-1)
            
        action = action.cpu().numpy()[0]
        # scale to action space if necessary; our network outputs in [-1,1] due to tanh already

        return action

    def extract_previous_action(self, state):
        """
        Extract the previous action from the given state.
        Supports both NumPy arrays and Torch tensors.
        Returns a copy (not a view) to avoid modifying the original state.
        """
        if isinstance(state, torch.Tensor):
            # Torch tensor: clone to avoid referencing original memory
            previous_action = state[:, -self.action_dim:].clone()
        elif isinstance(state, np.ndarray):
            # NumPy array: copy to avoid referencing original memory
            previous_action = state[:, -self.action_dim:].copy()
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

        return previous_action

    def update(self, replay: ReplayBuffer):
        cfg = self.config
        if len(replay) < cfg.batch_size:
            return {}
        B = cfg.batch_size
        states, noises, actions, rewards, next_states, next_noises, dones = replay.sample(cfg.batch_size)

        t = torch.linspace(self.dt, 1, self.steps-1, device=self.device).unsqueeze(0).repeat(B, 1)  # (B, steps)
        
        t_prev = torch.tensor(t - self.dt, device=self.device)  # (B, steps)
        t_next = torch.tensor(t + self.dt, device=self.device)  # (B, steps)

        

        # 2. start with states repeated for all timesteps
        obs = states.unsqueeze(1).repeat(1, self.steps-1, 1)  # (B, steps, state_dim)
        
        
        next_obs = obs.clone()
        # 3. replace the last timestep with next_states
        next_obs[:, -1, :] = next_states  # when t = 1, use next_state

        # 4. flatten for batch processing
        t_prev = t_prev.reshape(-1, 1)                    # (B * steps, 1)
        t = t.reshape(-1, 1)                              # (B * steps, 1)
        t_next = t_next.reshape(-1, 1)                    # (B * steps, 1)
        
        obs = obs.reshape(-1, states.shape[-1])  # (B * steps, state_dim)
        next_obs = next_obs.reshape(-1, states.shape[-1])  # (B * steps, state_dim)
        

        noises_exp = noises.unsqueeze(1).repeat(1, self.steps-1, 1).reshape(-1, noises.shape[-1])
        actions_exp = actions.unsqueeze(1).repeat(1, self.steps-1, 1).reshape(-1, actions.shape[-1])
        next_noises_exp = next_noises.unsqueeze(1).repeat(1, self.steps-1, 1).reshape(-1, next_noises.shape[-1])
        # print(noises_exp.shape,actions_exp.shape)
        # print(noises_prev_action.shape,noises_action.shape)
        noises_prev_action = torch.tensor((1 - t_prev) * noises_exp + t_prev * actions_exp, dtype=torch.float32, device=self.device)
        actions_current = torch.tensor((1 - t) * noises_exp + t * actions_exp, dtype=torch.float32, device=self.device)
        obs = torch.cat([obs, noises_prev_action], dim=-1)

        # actions_current = (1 - t) * noises + t * actions  # elementwise interpolation

        # if t_next > 1 → use next_noises instead
        noises_next_action = torch.where(
            (t_next > 1.0),  # condition
            next_noises_exp,     # if True → use next noise
            actions_current,           # if False → use interpolation
            )

        velocities = (actions - noises).unsqueeze(1).repeat(1, self.steps-1, 1)  # (B, steps-1, action_dim)
        velocities=velocities.reshape(-1, self.action_dim)  # (B * steps, action_dim)
        # print("velocities shape:",velocities.shape)
        with torch.no_grad():
            new_vel, log_prob, _ = self.policy(obs,t) # (B * steps, 1)
            
        rewards_flow = -(velocities- new_vel)**2  # (B*steps-1, action_dim)
        flow_reward_scalar = rewards_flow.mean(dim=-1, keepdim=True)  # (B* steps-1, 1)
        # print("flow_reward_scalar shape:",flow_reward_scalar.shape)
        rewards_expanded = torch.zeros((B, self.steps-1, 1), device=self.device)
        
        # rewards_expanded += flow_reward_scalar
        rewards_expanded[:, -1, :] = rewards  # only last timestep has reward
        
        dones_expanded = torch.zeros((B, self.steps-1, 1), device=self.device)
        dones_expanded[:, -1, :] = dones    # only last timestep has done
        rewards_flat = rewards_expanded.reshape(-1, 1)    # (B * steps, 1)
        rewards_flat += 0.0001 * flow_reward_scalar
        dones_flat = dones_expanded.reshape(-1, 1)        # (B * steps, 1)
        
        next_obs = torch.cat([next_obs, noises_next_action], dim=-1)
        # print("nprev ext_obs shape after concat:",next_obs.shape)
        t_next = torch.where(t_next > 1.0, torch.zeros_like(t_next), t_next)    
        
        with torch.no_grad():
            # sample next action from policy, compute log prob
            # print("next_obs shape:",next_obs.shape)
            next_vel, next_log_prob, _ = self.policy(next_obs,t_next)

            # scale next_action to env action space for Q calculation (critics expect env scaled? We use unscaled consistently)
            # We will keep Q networks consistent with unscaled [-1,1] action representation
            # compute target Q value using targets
            # prev_action = self.extract_previous_action(next_states)
            next_action = next_vel*self.dt + noises_next_action

            next_action = torch.clamp(torch.tensor(next_action, dtype=torch.float32, device=self.device),-1,1)

            q1_next_target = self.q1_target(next_obs, next_action,t_next)
            q2_next_target = self.q2_target(next_obs, next_action,t_next)
            q_next_target = torch.min(q1_next_target, q2_next_target) - (self.alpha.detach() * next_log_prob)
            
            # gamma_mask = torch.where(torch.isclose(t, torch.tensor(1.0, device=self.device)), cfg.gamma, 1.0)
            # target_q = rewards + (1.0 - dones) * gamma_mask * q_next_target
            target_q = rewards_flat + (1.0 - dones_flat) * cfg.gamma * q_next_target

        # Critic losses
        # print("obs shape:",obs.shape)
        q1_pred = self.q1(obs, actions_current,t)
        q2_pred = self.q2(obs, actions_current,t)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optim.zero_grad(); q1_loss.backward(); self.q1_optim.step()
        self.q2_optim.zero_grad(); q2_loss.backward(); self.q2_optim.step()

        # Policy loss (sampled from states)
        new_vel, log_prob, _ = self.policy(obs,t)
        # prev_action = self.extract_previous_action(states)
        new_action = new_vel*self.dt + noises_prev_action

        new_action = torch.clamp(new_action,-1,1)

        q1_new = self.q1(obs, new_action,t)
        q2_new = self.q2(obs, new_action,t)
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
