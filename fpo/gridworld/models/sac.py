import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from torch.distributions import MultivariateNormal

import time
import wandb

class SAC:
    """
    This is the SAC class we will use as our model in main.py
    """
    def __init__(self, policy_class, env, **hyperparameters):
        """
        Initializes the SAC model, including hyperparameters.

        Parameters:
            policy_class - the policy class to use for our actor and critic networks.
            env - the environment to train on.
            hyperparameters - all extra arguments passed into SAC that should be hyperparameters.
        """
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

        self._init_hyperparameters(hyperparameters)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Environment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.action_scale = torch.tensor(env.action_space.high, device=self.device)
        self.action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, device=self.device)

        # Initialize networks
        self.actor = policy_class(self.obs_dim, self.act_dim).to(self.device)   # ALG STEP 1
        # self.critic = policy_class(self.obs_dim, 1).to(self.device)
        # self.actor = policy_class(self.obs_dim, self.act_dim, is_actor=True).to(self.device)
        self.critic1 = policy_class(self.obs_dim, 1).to(self.device)
        self.critic2 = policy_class(self.obs_dim, 1).to(self.device)
        self.target_critic1 = policy_class(self.obs_dim, 1).to(self.device)
        self.target_critic2 = policy_class(self.obs_dim, 1).to(self.device)
        
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.log_alpha = torch.tensor(np.log(self.alpha_init), requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.alpha_lr)

        # Replay buffer
        self.replay_buffer = []
        self.max_buffer_size = self.buffer_size

        # Logging
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'actor_losses': [],
            'critic_losses': [],
            'alpha_losses': [],
            'batch_rews': [],
            'batch_lens': [],
        }

    def learn(self, total_timesteps):
        """
        Main SAC training loop.
        """
        print(f"Learning... total timesteps: {total_timesteps}")
        t_so_far = 0
        i_so_far = 0
        obs, _ = self.env.reset()

        while t_so_far < total_timesteps:
            # === Rollout ===
            batch_rews, batch_lens = self.rollout()
            t_so_far += sum(batch_lens)
            i_so_far += 1
            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # === Learn ===
            if len(self.replay_buffer) >= self.batch_size:
                for _ in range(self.updates_per_iter):
                    self.update()

            # === Logging ===
            self._log_summary()

            # Save models
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './sac_actor.pth')
                torch.save(self.critic1.state_dict(), './sac_critic1.pth')
                torch.save(self.critic2.state_dict(), './sac_critic2.pth')
                wandb.save(f"{self.run_name}_sac_actor_iter{i_so_far}.pth")

    def rollout(self):
        """
        Collects experience from the environment using the current policy.
        """
        batch_rews = []
        batch_lens = []

        for _ in range(self.timesteps_per_batch):
            episode_rews = []
            obs, _ = self.env.reset()
            done = False
            t = 0

            while not done and t < self.max_timesteps_per_episode:
                action, _ = self.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.store_transition(obs, action, reward, next_obs, done)
                episode_rews.append(reward)
                obs = next_obs
                t += 1

            batch_rews.append(sum(episode_rews))
            batch_lens.append(t)

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
        return batch_rews, batch_lens

    def store_transition(self, obs, act, rew, next_obs, done):
        if len(self.replay_buffer) >= self.max_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((obs, act, rew, next_obs, done))

    def sample_batch(self):
        idx = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in idx]
        obs, acts, rews, next_obs, dones = map(np.stack, zip(*batch))
        return (
            torch.tensor(obs, dtype=torch.float, device=self.device),
            torch.tensor(acts, dtype=torch.float, device=self.device),
            torch.tensor(rews, dtype=torch.float, device=self.device).unsqueeze(1),
            torch.tensor(next_obs, dtype=torch.float, device=self.device),
            torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1),
        )

    def update(self):
        obs, acts, rews, next_obs, dones = self.sample_batch()

        # === Sample next actions from policy ===
        next_actions, next_log_probs = self.get_action(next_obs)
        next_actions = torch.tensor(next_actions).to(self.device)
        with torch.no_grad():
            target_q1 = self.target_critic1(torch.cat([next_obs, next_actions], dim=-1))
            target_q2 = self.target_critic2(torch.cat([next_obs, next_actions], dim=-1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_value = rews + self.gamma * (1 - dones) * target_q

        # === Critic update ===
        current_q1 = self.critic1(torch.cat([obs, acts], dim=-1))
        current_q2 = self.critic2(torch.cat([obs, acts], dim=-1))
        critic1_loss = F.mse_loss(current_q1, target_value)
        critic2_loss = F.mse_loss(current_q2, target_value)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        critic_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()

        # === Actor update ===
        new_actions, log_probs = self.actor.get_action(obs)
        q1_new = self.critic1(torch.cat([obs, new_actions], dim=-1))
        q2_new = self.critic2(torch.cat([obs, new_actions], dim=-1))
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # === Alpha (entropy temperature) update ===
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # === Soft target update ===
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        # Logging
        self.logger['actor_losses'].append(actor_loss.item())
        self.logger['critic_losses'].append(critic_loss.item())
        self.logger['alpha_losses'].append(alpha_loss.item())

    def get_action(self, obs, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        mean = self.actor(obs_tensor)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)
        # action, _ = self.actor.sample(obs_tensor, deterministic)
        action = action.squeeze(0).detach().cpu().numpy()
        return np.clip(action, -2, 2), log_prob

    def _soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _init_hyperparameters(self, hyperparameters):
        """Initialize default and custom hyperparameters."""
        self.gamma = 0.99
        self.tau = 0.005
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.alpha_lr = 3e-4
        self.alpha_init = 0.2
        self.buffer_size = int(1e6)
        self.batch_size = 256
        self.updates_per_iter = 1
        self.target_entropy = -1.0
        self.timesteps_per_batch = 1000
        self.max_timesteps_per_episode = 1000
        self.save_freq = 10
        self.run_name = "unnamed_sac_run"

        for param, val in hyperparameters.items():
            setattr(self, param, val)

    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_rew = np.mean(self.logger['batch_rews']) if self.logger['batch_rews'] else 0
        avg_actor_loss = np.mean(self.logger['actor_losses']) if self.logger['actor_losses'] else 0

        print(f"-------------------- Iteration #{i_so_far} --------------------")
        print(f"Timesteps So Far: {t_so_far}")
        print(f"Avg Return: {avg_rew:.2f}")
        print(f"Avg Actor Loss: {avg_actor_loss:.5f}")
        print(f"Iteration took: {delta_t:.2f} secs")
        print("------------------------------------------------------", flush=True)

        wandb.log({
            "iteration": i_so_far,
            "timesteps_so_far": t_so_far,
            "avg_episode_return": avg_rew,
            "avg_actor_loss": avg_actor_loss,
            "iteration_duration_sec": delta_t,
        })

        self.logger['batch_rews'] = []
        self.logger['batch_lens'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
        self.logger['alpha_losses'] = []
