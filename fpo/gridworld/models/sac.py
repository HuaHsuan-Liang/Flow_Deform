import gymnasium as gym
import time
import wandb
import argparse
import math
import os
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Tuple, Optional, Deque

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class SAC_self:
    """
    Soft Actor-Critic implementation matching the PPO API.
    """
    def __init__(self, policy_class, env, **hyperparameters):
        # Environment check
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Box
        
        # Initialize hyperparameters
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Environment info
        self.env = env
        self._init_hyperparameters(hyperparameters)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        # Actor & Critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim).to(self.device)
        self.critic1 = policy_class(self.obs_dim + self.act_dim, 1).to(self.device)
        self.critic2 = policy_class(self.obs_dim + self.act_dim, 1).to(self.device)
        self.critic1_target = policy_class(self.obs_dim + self.act_dim, 1).to(self.device)
        self.critic2_target = policy_class(self.obs_dim + self.act_dim, 1).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(500_000, torch.device(self.device))

        # Logger
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rews': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': []
        }

    def learn(self, total_timesteps):
        print(f"Learning SAC... Running up to {total_timesteps} timesteps")
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        
        t_so_far = 0
        i_so_far = 0

        while t_so_far < total_timesteps:
            
            t_so_far += self.rollout()
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far
            loss_actor = []
            loss_critic = []
            loss_alpha = []
            for _ in range(self.n_updates_per_iteration):  
                batch_obs, batch_acts, batch_rews, batch_next_obs, batch_dones = self.replay_buffer.sample(self.batch_size)
                # Convert batch to tensors
                obs = torch.tensor(batch_obs, dtype=torch.float32).to(self.device)
                acts = torch.tensor(batch_acts, dtype=torch.float32).to(self.device)
                rews = torch.tensor(batch_rews, dtype=torch.float32).unsqueeze(-1).to(self.device)
                next_obs = torch.tensor(batch_next_obs, dtype=torch.float32).to(self.device)
                dones = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

                # === Critic update ===
                with torch.no_grad():
                    next_actions, next_log_probs = self.get_action(next_obs)
                    next_actions = torch.tensor(next_actions, dtype=torch.float32).to(self.device)
                    next_log_probs = torch.tensor(next_log_probs, dtype=torch.float32).to(self.device)
                    
                    q1_next = self.critic1_target(torch.cat([next_obs, next_actions], dim=-1))
                    q2_next = self.critic2_target(torch.cat([next_obs, next_actions], dim=-1))
                    q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_probs
                    target_q = rews + self.gamma * (1 - dones) * q_next

                # Critic losses
                q1 = self.critic1(torch.cat([obs, acts], dim=-1))
                q2 = self.critic2(torch.cat([obs, acts], dim=-1))
                critic1_loss = nn.functional.mse_loss(q1, target_q)
                critic2_loss = nn.functional.mse_loss(q2, target_q)

                self.critic1_optim.zero_grad()
                critic1_loss.backward()
                self.critic1_optim.step()

                self.critic2_optim.zero_grad()
                critic2_loss.backward()
                self.critic2_optim.step()

                # === Actor update ===
                actions_pred, log_probs = self.forward(obs)
                actions_pred = torch.tensor(actions_pred, dtype=torch.float32).to(self.device)
                log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device)
                q1_pi = self.critic1(torch.cat([obs, actions_pred], dim=-1))
                q2_pi = self.critic2(torch.cat([obs, actions_pred], dim=-1))
                actor_loss = (self.alpha.detach() * log_probs - torch.min(q1_pi, q2_pi)).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                alpha_loss = torch.tensor(0.0, device=self.device)
                if self.log_alpha is not None:
                    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    
                loss_actor.append(actor_loss.detach())
                loss_critic.append(((critic1_loss + critic2_loss)/2).detach())
                loss_alpha.append(alpha_loss.detach())  
                # === Soft update target networks ===
                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Logging
            self.logger['actor_losses'].append(sum(loss_actor) / len(loss_actor))
            self.logger['critic_losses'].append(sum(loss_critic) / len(loss_critic))
            self.logger['alpha_losses'].append(sum(loss_alpha) / len(loss_alpha))
            self._log_summary()

            # Save
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './sac_actor.pth')
                torch.save(self.critic1.state_dict(), './sac_critic1.pth')
                torch.save(self.critic2.state_dict(), './sac_critic2.pth')
                wandb.save(f"{self.run_name}_actor_iter{i_so_far}.pth")
                wandb.save(f"{self.run_name}_critic1_iter{i_so_far}.pth")
                wandb.save(f"{self.run_name}_critic2_iter{i_so_far}.pth")

    def rollout(self):
        count =0
        obs, _ = self.env.reset()
        done = False
        ep_len = 0
        ep_rews = 0

        while count < self.timesteps_per_batch:
            if (self.logger['t_so_far']<= 5000):
                action = self.env.action_space.sample()
            else:
                action, _ = self.get_action(obs)
                
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated | truncated

            count+=1
            self.replay_buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            ep_len += 1
            ep_rews += reward

            if done or ep_len >= self.max_timesteps_per_episode:
                obs, _ = self.env.reset()
                done = False
                self.logger['batch_lens'].append(ep_len)
                self.logger['batch_rews'].append(ep_rews)
                ep_len = 0
                ep_rews = 0

        return count

    def get_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
        mean = self.actor(obs_tensor)
        mean = torch.tanh(mean)  # Ensure actions are in [-1, 1]
        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return self._rescale_action(action.detach().cpu().numpy()), log_prob.detach().cpu()
    
    def forward(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
        mean = self.actor(obs_tensor)
        mean = torch.tanh(mean)  # Ensure actions are in [-1, 1]
        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return self._rescale_action(action.cpu()), log_prob.cpu()

    def _rescale_action(self, action_unscaled):
        """
        Map action from [-1, 1] to [low, high].
        Works with both NumPy arrays and PyTorch tensors.
        """
        low, high = self.action_low, self.action_high

        if isinstance(action_unscaled, torch.Tensor):
            low = torch.as_tensor(low, dtype=action_unscaled.dtype, device=action_unscaled.device)
            high = torch.as_tensor(high, dtype=action_unscaled.dtype, device=action_unscaled.device)
            return low + 0.5 * (action_unscaled + 1.0) * (high - low)
        else:  # NumPy
            return low + 0.5 * (action_unscaled + 1.0) * (high - low)


    def _inv_rescale_action(self, action):
        """
        Map action from [low, high] back to [-1, 1].
        Works with both NumPy arrays and PyTorch tensors.
        """
        low, high = self.action_low, self.action_high

        if isinstance(action, torch.Tensor):
            low = torch.as_tensor(low, dtype=action.dtype, device=action.device)
            high = torch.as_tensor(high, dtype=action.dtype, device=action.device)
            return 2.0 * (action - low) / (high - low) - 1.0
        else:  # NumPy
            return 2.0 * (action - low) / (high - low) - 1.0
    
    def _init_hyperparameters(self, hyperparameters):
        # Default hyperparameters
        self.lr = 3e-4
        self.gamma = 0.99
        self.alpha = 0.2       # Entropy coefficient
        self.tau = 0.005       # Soft update rate
        self.timesteps_per_batch = 4000
        self.max_timesteps_per_episode = 1000
        self.n_updates_per_iteration = 5
        self.save_freq = 10
        self.render = False
        self.run_name = "sac_run"
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.target_entropy = -float(self.env.action_space.shape[0])  # Target entropy heuristic
        # log alpha parameter
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.batch_size = 256
        # Update with custom hyperparameters
        for param, val in hyperparameters.items():
            setattr(self, param, val)

    @property
    def alpha(self):
        if self.log_alpha is None:
            return self._alpha_fixed
        else:
            return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, value):
        self._alpha_fixed = value
        
    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9

        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean(self.logger['batch_rews'])
        avg_actor_loss = np.mean([loss.float().cpu().item() for loss in self.logger['actor_losses']])
        avg_critic_loss = np.mean([loss.float().cpu().item() for loss in self.logger['critic_losses']])
        avg_alpha_loss = np.mean([loss.float().cpu().item() for loss in self.logger['alpha_losses']])

        print(flush=True)
        print(f"-------------------- Iteration #{self.logger['i_so_far']} --------------------")
        print(f"Average Episodic Length: {avg_ep_lens:.2f}")
        print(f"Average Episodic Return: {avg_ep_rews:.2f}")
        print(f"Average Actor Loss: {avg_actor_loss:.5f}")
        print(f"Average Critic Loss: {avg_critic_loss:.5f}")
        print(f"Average Alpha Loss: {avg_alpha_loss:.5f}")
        print(f"Iteration took: {delta_t:.2f} secs")
        print("------------------------------------------------------", flush=True)

        wandb.log({
            "iteration": self.logger['i_so_far'],
            "avg_episode_length": float(avg_ep_lens),
            "avg_episode_return": float(avg_ep_rews),
            "avg_actor_loss": float(avg_actor_loss),
            "avg_critic_loss": float(avg_critic_loss),
            "avg_alpha_loss": float(avg_alpha_loss),
            "iteration_duration_sec": float(delta_t)
        })

        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
        self.logger['alpha_losses'] = []


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