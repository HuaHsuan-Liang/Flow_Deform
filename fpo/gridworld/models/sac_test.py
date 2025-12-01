import time
import random
from collections import deque, namedtuple
from typing import Deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
import wandb


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
        states = torch.tensor(
            np.array([b.state for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            np.array([b.action for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor(
            np.array([b.reward for b in batch]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.array([b.next_state for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            np.array([b.done for b in batch]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class SAC_self:
    """
    Soft Actor-Critic implementation matching the PPO-style API 
    Uses:
      * tanh-squashed Gaussian actor
      * twin Q-networks + target networks
      * automatic entropy tuning (alpha)
    """

    def __init__(self, policy_class, env, eval_env, **hyperparameters):
        # Env checks
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        self.env = env
        self.eval_env = eval_env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)

        # Hyperparameters
        self._init_hyperparameters(hyperparameters)

        # Actor: policy_class provides the mean; log_std is a separate learnable vector
        self.actor = policy_class(self.obs_dim, self.act_dim).to(self.device)
        # one log_std per action dimension, shared across states
        self.log_std = nn.Parameter(torch.zeros(self.act_dim, device=self.device))

        # Critics: Q(s,a) networks + targets
        self.critic1 = policy_class(self.obs_dim + self.act_dim, 1).to(self.device)
        self.critic2 = policy_class(self.obs_dim + self.act_dim, 1).to(self.device)
        self.critic1_target = policy_class(self.obs_dim + self.act_dim, 1).to(self.device)
        self.critic2_target = policy_class(self.obs_dim + self.act_dim, 1).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optim = Adam(list(self.actor.parameters()) + [self.log_std], lr=self.lr)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=self.lr)

        # Entropy / alpha
        if self.auto_entropy_tuning:
            # log_alpha is the learnable parameter
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
        else:
            self.log_alpha = None
            self._alpha_fixed = torch.tensor(self.alpha_fixed, device=self.device)
            self.alpha_optim = None

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_capacity, self.device)

        # Logger
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rews': [],
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
        }

        # Best model tracking (by eval mean return)
        self.best_mean_ret = -float("inf")

    # Hyperparameters / alpha
    def _init_hyperparameters(self, hyperparameters):
        # Core SAC params
        self.lr = hyperparameters.get('lr', 3e-4)
        self.gamma = hyperparameters.get('gamma', 0.99)
        self.tau = hyperparameters.get('tau', 0.005)

        # Data collection / updates
        self.timesteps_per_batch = hyperparameters.get('timesteps_per_batch', 4096)
        self.max_timesteps_per_episode = hyperparameters.get('max_timesteps_per_episode', 1000)
        self.n_updates_per_iteration = hyperparameters.get('n_updates_per_iteration', 5)
        self.batch_size = hyperparameters.get('batch_size', 256)
        self.replay_capacity = hyperparameters.get('replay_capacity', 500_000)

        # Entropy
        self.auto_entropy_tuning = hyperparameters.get('auto_entropy_tuning', True)
        self.alpha_fixed = hyperparameters.get('alpha', 0.2)
        self.target_entropy = hyperparameters.get(
            'target_entropy',
            -float(self.act_dim)  # standard heuristic
        )

        # Logging / saving
        self.save_freq = hyperparameters.get('save_freq', 10)
        self.render = hyperparameters.get('render', False)
        self.run_name = hyperparameters.get('run_name', 'sac_run')

    @property
    def alpha(self):
        if self.log_alpha is None:
            return self._alpha_fixed
        else:
            return self.log_alpha.exp()

    # Public training API
    def learn(self, total_timesteps: int):
        print(f"Learning SAC... Running up to {total_timesteps} timesteps")
        print(f"{self.timesteps_per_batch} timesteps per batch")

        t_so_far = 0
        i_so_far = 0

        while t_so_far < total_timesteps:
            collected = self.rollout()
            t_so_far += collected
            i_so_far += 1

            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # If we don't have enough data yet, skip updates
            if len(self.replay_buffer) < self.batch_size:
                self._log_summary()  # still log episodic stats
                continue

            actor_losses = []
            critic_losses = []
            alpha_losses = []

            for _ in range(self.n_updates_per_iteration):
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

                # Critic update
                with torch.no_grad():
                    next_actions, next_log_probs = self._sample_action_and_log_prob(next_states)
                    q1_next = self.critic1_target(torch.cat([next_states, next_actions], dim=-1))
                    q2_next = self.critic2_target(torch.cat([next_states, next_actions], dim=-1))
                    q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_probs
                    target_q = rewards + self.gamma * (1.0 - dones) * q_next

                q1 = self.critic1(torch.cat([states, actions], dim=-1))
                q2 = self.critic2(torch.cat([states, actions], dim=-1))

                critic1_loss = nn.functional.mse_loss(q1, target_q)
                critic2_loss = nn.functional.mse_loss(q2, target_q)
                critic_loss = critic1_loss + critic2_loss

                self.critic1_optim.zero_grad()
                self.critic2_optim.zero_grad()
                critic_loss.backward()
                self.critic1_optim.step()
                self.critic2_optim.step()

                # Actor update 
                new_actions, log_probs = self._sample_action_and_log_prob(states)
                q1_pi = self.critic1(torch.cat([states, new_actions], dim=-1))
                q2_pi = self.critic2(torch.cat([states, new_actions], dim=-1))
                q_pi = torch.min(q1_pi, q2_pi)

                actor_loss = (self.alpha.detach() * log_probs - q_pi).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Alpha / entropy update
                if self.auto_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                else:
                    alpha_loss = torch.zeros(1, device=self.device)

                # soft update targets
                self._soft_update(self.critic1, self.critic1_target)
                self._soft_update(self.critic2, self.critic2_target)

                actor_losses.append(actor_loss.detach())
                critic_losses.append(critic_loss.detach())
                alpha_losses.append(alpha_loss.detach())

            self.logger['actor_losses'].append(torch.stack(actor_losses).mean())
            self.logger['critic_losses'].append(torch.stack(critic_losses).mean())
            self.logger['alpha_losses'].append(torch.stack(alpha_losses).mean())

            self._log_summary()

            # Periodic eval + save
            if i_so_far % self.save_freq == 0:
                mean_ret, std_ret = self.evaluate_policy(self.eval_env, episodes=3)
                wandb.log({
                    "eval/mean": mean_ret,
                    "eval/std": std_ret,
                    "alpha": float(self.alpha.detach().cpu()),
                }, step=self.logger['i_so_far'])

                torch.save(self.actor.state_dict(), './sac_actor.pth')
                torch.save(self.critic1.state_dict(), './sac_critic1.pth')
                torch.save(self.critic2.state_dict(), './sac_critic2.pth')
                wandb.save(f"{self.run_name}_actor_iter{i_so_far}.pth")
                wandb.save(f"{self.run_name}_critic1_iter{i_so_far}.pth")
                wandb.save(f"{self.run_name}_critic2_iter{i_so_far}.pth")

                # Save best checkpoints based on eval mean return
                if mean_ret > self.best_mean_ret:
                    self.best_mean_ret = mean_ret
                    best_actor_path = f"{self.run_name}_best_actor.pth"
                    best_c1_path = f"{self.run_name}_best_critic1.pth"
                    best_c2_path = f"{self.run_name}_best_critic2.pth"
                    torch.save(self.actor.state_dict(), best_actor_path)
                    torch.save(self.critic1.state_dict(), best_c1_path)
                    torch.save(self.critic2.state_dict(), best_c2_path)
                    wandb.summary["best_eval_mean_return"] = mean_ret
                    wandb.save(best_actor_path)
                    wandb.save(best_c1_path)
                    wandb.save(best_c2_path)

    # Rollout & evaluation
    def rollout(self) -> int:
        """
        Collect timesteps_per_batch steps into the replay buffer.
        """
        count = 0
        obs, _ = self.env.reset()
        done = False
        ep_len = 0
        ep_rews = 0.0

        while count < self.timesteps_per_batch:
            if self.logger['t_so_far'] <= 5_000:
                action = self.env.action_space.sample()
            else:
                action, _ = self.get_action(obs)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.push(obs, action, reward, next_obs, float(done))

            count += 1
            obs = next_obs
            ep_len += 1
            ep_rews += reward

            if done or ep_len >= self.max_timesteps_per_episode:
                self.logger['batch_lens'].append(ep_len)
                self.logger['batch_rews'].append(ep_rews)

                obs, _ = self.env.reset()
                done = False
                ep_len = 0
                ep_rews = 0.0

        return count

    def evaluate_policy(self, env, episodes: int = 5, max_episode_steps=None, render=False):
        returns = []

        for _ in range(episodes):
            s, _ = env.reset()
            done = False
            ep_ret = 0.0
            steps = 0

            while not done:
                action, _ = self.get_action(s, deterministic=True)
                s, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_ret += reward
                steps += 1

                if max_episode_steps and steps >= max_episode_steps:
                    break
                if render:
                    env.render()

            returns.append(ep_ret)

        return float(np.mean(returns)), float(np.std(returns))


    # Acting (for env interaction)
    def get_action(self, obs, deterministic: bool = False):
        """
        obs: single observation (np array or tensor) or batch.
        Returns: (action_np, log_prob_np_or_None)
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        if deterministic:
            mean = self.actor(obs_tensor)
            u = torch.tanh(mean)
            log_prob = None
        else:
            mean = self.actor(obs_tensor)
            std = self.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            z = dist.rsample()  # reparam
            u = torch.tanh(z)
            # Tanh correction
            log_prob = dist.log_prob(z) - torch.log(1 - u.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Map from [-1,1] to [low, high] (Jacobian is constant, so we ignore it)
        action_scaled = self._rescale_action(u)

        action_np = action_scaled.detach().cpu().numpy()
        if action_np.shape[0] == 1:
            action_np = action_np[0]

        if log_prob is None:
            return action_np, None
        else:
            log_prob_np = log_prob.detach().cpu().numpy()
            if log_prob_np.shape[0] == 1:
                log_prob_np = log_prob_np[0]
            return action_np, log_prob_np


    def _sample_action_and_log_prob(self, obs_tensor: torch.Tensor):
        """
        Same as get_action but stays purely in torch-land and returns tensors.
        obs_tensor: (B, obs_dim)
        """
        mean = self.actor(obs_tensor)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        z = dist.rsample()
        u = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - u.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        action_scaled = self._rescale_action(u)
        return action_scaled, log_prob

    def _rescale_action(self, action_unscaled: torch.Tensor) -> torch.Tensor:
        """
        Map action from [-1, 1] to [low, high].
        Expects torch tensor.
        """
        # broadcasting over batch dimension
        return self.action_low + 0.5 * (action_unscaled + 1.0) * (self.action_high - self.action_low)

    def _soft_update(self, source_net: nn.Module, target_net: nn.Module):
        for param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    # logging 
    def _log_summary(self):
        delta_t_prev = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t_prev) / 1e9

        avg_ep_lens = np.mean(self.logger['batch_lens']) if self.logger['batch_lens'] else 0.0
        avg_ep_rews = np.mean(self.logger['batch_rews']) if self.logger['batch_rews'] else 0.0

        def _mean_loss(key):
            if not self.logger[key]:
                return 0.0
            vals = [x.float().cpu().item() for x in self.logger[key]]
            return float(np.mean(vals))

        avg_actor_loss = _mean_loss('actor_losses')
        avg_critic_loss = _mean_loss('critic_losses')
        avg_alpha_loss = _mean_loss('alpha_losses')

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
            "timesteps_so_far": self.logger['t_so_far'],
            "avg_episode_length": float(avg_ep_lens),
            "avg_episode_return": float(avg_ep_rews),
            "avg_actor_loss": float(avg_actor_loss),
            "avg_critic_loss": float(avg_critic_loss),
            "avg_alpha_loss": float(avg_alpha_loss),
            "alpha": float(self.alpha.detach().cpu()),
            "iteration_duration_sec": float(delta_t),
        }, step=self.logger['i_so_far'])

        # reset per-iteration buffers
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
        self.logger['alpha_losses'] = []
