"""
Refactored SAC implementation matching the working module_fsac/SAC architecture.
Key changes from original sac_test.py:
1. Larger hidden layers (256x256 instead of 64x64)
2. State-dependent log_std (separate network head)
3. Proper weight initialization (orthogonal)
4. Proper Gaussian policy with mean/log_std heads
5. Video recording during evaluation
"""

import time
import random
import math
import os
from collections import deque, namedtuple
from typing import Deque, Tuple, List, Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
import wandb


# Utility Functions & Classes

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


def init_weights(m):
    """Orthogonal initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


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

# Network Classes

class MLP(nn.Module):
    """Standard MLP with orthogonal initialization."""
    def __init__(self, inp_dim: int, out_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        last = inp_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.model = nn.Sequential(*layers)
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class QNetwork(nn.Module):
    """Q-network: Q(s, a) -> scalar value."""
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1, hidden_sizes)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q(x)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy with reparameterization and tanh squashing.
    Returns action, log_prob, and mean (pre-tanh mean).
    
    Key difference from original: state-dependent log_std via separate head.
    """
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_sizes: Tuple[int, ...] = (256, 256),
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim

        # Shared trunk
        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.net = nn.Sequential(*layers)
        
        # Separate heads for mean and log_std
        self.mean_linear = nn.Linear(last, action_dim)
        self.log_std_linear = nn.Linear(last, action_dim)
        
        # Apply orthogonal init to trunk
        self.net.apply(init_weights)
        
        # Small uniform init for output layers (helps stability)
        nn.init.uniform_(self.mean_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_linear.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_linear.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_linear.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        """
        Args:
            state: (B, state_dim) tensor
            deterministic: if True, return tanh(mean) without sampling
            
        Returns:
            action: (B, action_dim) in [-1, 1]
            log_prob: (B, 1) or None if deterministic
            mean: (B, action_dim) pre-tanh mean
        """
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            return action, None, mean

        # Reparameterization trick
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        # Log prob with tanh squashing correction
        # log π(a|s) = log π(z|s) - sum(log(1 - tanh²(z)))
        log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)
        # Numerically stable correction
        eps = 1e-6
        log_prob -= torch.log(1 - action.pow(2) + eps).sum(dim=-1, keepdim=True)
        
        return action, log_prob, mean

# Video Recording Utilities

def save_video(frames: List, path: str, fps: int = 30):
    """
    Save frames to an MP4 video file.
    
    Args:
        frames: List of frames (numpy arrays, HWC format) or list of episode frame lists
        path: Output path for video
        fps: Frames per second
    """
    if not frames:
        return
    
    # Flatten if nested (list of episodes)
    if isinstance(frames[0], list):
        frames = [f for ep_frames in frames for f in ep_frames]
    
    if not frames:
        return
        
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    
    for frame in frames:
        # Convert RGB to BGR if needed (gymnasium returns RGB)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        out.write(frame_bgr)
    
    out.release()
    print(f"Video saved to {path}")

# SAC Agent

class SAC_self:
    """
    Soft Actor-Critic implementation matching the PPO-style API.
    
    Uses:
      * Proper GaussianPolicy with state-dependent std
      * Twin Q-networks + target networks (256x256 hidden)
      * Automatic entropy tuning (alpha)
      * Orthogonal weight initialization
      * Video recording during evaluation
    """

    def __init__(self, policy_class, env, eval_env, **hyperparameters):
        """
        Args:
            policy_class: Ignored - we use our own GaussianPolicy and QNetwork
            env: Training environment
            eval_env: Evaluation environment
            **hyperparameters: SAC hyperparameters
        """
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

        # Initialize hyperparameters first
        self._init_hyperparameters(hyperparameters)

        # Networks with proper architecture
        hidden_sizes = self.hidden_sizes
        
        # Actor: Gaussian policy with state-dependent std
        self.actor = GaussianPolicy(
            self.obs_dim, self.act_dim, 
            hidden_sizes=hidden_sizes
        ).to(self.device)
        
        # Critics: Q(s,a) networks + targets
        self.critic1 = QNetwork(self.obs_dim, self.act_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.critic2 = QNetwork(self.obs_dim, self.act_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.critic1_target = QNetwork(self.obs_dim, self.act_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.critic2_target = QNetwork(self.obs_dim, self.act_dim, hidden_sizes=hidden_sizes).to(self.device)

        # Copy weights to targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=self.lr)

        # Automatic entropy tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -float(self.act_dim)  # Heuristic: -|A|
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
        else:
            self.log_alpha = None
            self._alpha_fixed = torch.tensor(self.alpha_fixed, device=self.device)
            self.alpha_optim = None
            self.target_entropy = None

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

    def _init_hyperparameters(self, hyperparameters):
        """Initialize hyperparameters with sensible defaults."""
        # Core SAC params
        self.lr = hyperparameters.get('lr', 3e-4)
        self.gamma = hyperparameters.get('gamma', 0.99)
        self.tau = hyperparameters.get('tau', 0.005)

        # Network architecture
        self.hidden_sizes = hyperparameters.get('hidden_sizes', (256, 256))

        # Data collection / updates
        self.timesteps_per_batch = hyperparameters.get('timesteps_per_batch', 4096)
        self.max_timesteps_per_episode = hyperparameters.get('max_timesteps_per_episode', 1000)
        self.n_updates_per_iteration = hyperparameters.get('n_updates_per_iteration', 5)
        self.batch_size = hyperparameters.get('batch_size', 256)
        self.replay_capacity = hyperparameters.get('replay_capacity', 500_000)
        self.start_steps = hyperparameters.get('start_steps', 5000)  # Random exploration steps

        # Entropy
        self.auto_entropy_tuning = hyperparameters.get('auto_entropy_tuning', True)
        self.alpha_fixed = hyperparameters.get('alpha', 0.2)

        # Logging / saving
        self.save_freq = hyperparameters.get('save_freq', 10)
        self.video_freq = hyperparameters.get('video_freq', 50)  # Record video every N evals
        self.render = hyperparameters.get('render', False)
        self.run_name = hyperparameters.get('run_name', 'sac_run')

    @property
    def alpha(self):
        if self.log_alpha is None:
            return self._alpha_fixed
        else:
            return self.log_alpha.exp()

    # Training

    def learn(self, total_timesteps: int):
        print(f"Learning SAC... Running up to {total_timesteps} timesteps")
        print(f"{self.timesteps_per_batch} timesteps per batch")
        print(f"Using hidden sizes: {self.hidden_sizes}")

        t_so_far = 0
        i_so_far = 0

        while t_so_far < total_timesteps:
            collected = self.rollout()
            t_so_far += collected
            i_so_far += 1

            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Skip updates if not enough data
            if len(self.replay_buffer) < self.batch_size:
                self._log_summary()
                continue

            actor_losses = []
            critic_losses = []
            alpha_losses = []

            for _ in range(self.n_updates_per_iteration):
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

                # ============ Critic Update ============
                with torch.no_grad():
                    # Sample next actions from current policy
                    next_actions, next_log_probs, _ = self.actor(next_states)
                    
                    # Compute target Q values
                    q1_next = self.critic1_target(next_states, next_actions)
                    q2_next = self.critic2_target(next_states, next_actions)
                    q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_probs
                    target_q = rewards + self.gamma * (1.0 - dones) * q_next

                # Current Q estimates
                q1 = self.critic1(states, actions)
                q2 = self.critic2(states, actions)

                # Critic losses
                critic1_loss = F.mse_loss(q1, target_q)
                critic2_loss = F.mse_loss(q2, target_q)
                critic_loss = critic1_loss + critic2_loss

                self.critic1_optim.zero_grad()
                self.critic2_optim.zero_grad()
                critic_loss.backward()
                self.critic1_optim.step()
                self.critic2_optim.step()

                # ============ Actor Update ============
                new_actions, log_probs, _ = self.actor(states)
                q1_pi = self.critic1(states, new_actions)
                q2_pi = self.critic2(states, new_actions)
                q_pi = torch.min(q1_pi, q2_pi)

                # Actor loss: maximize Q - alpha * log_prob
                actor_loss = (self.alpha.detach() * log_probs - q_pi).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # ============ Alpha Update ============
                if self.auto_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                else:
                    alpha_loss = torch.zeros(1, device=self.device)

                # ============ Soft Update Targets ============
                self._soft_update(self.critic1, self.critic1_target)
                self._soft_update(self.critic2, self.critic2_target)

                actor_losses.append(actor_loss.detach())
                critic_losses.append(critic_loss.detach())
                alpha_losses.append(alpha_loss.detach())

            # Log losses
            self.logger['actor_losses'].append(torch.stack(actor_losses).mean())
            self.logger['critic_losses'].append(torch.stack(critic_losses).mean())
            self.logger['alpha_losses'].append(torch.stack(alpha_losses).mean())

            self._log_summary()

            # Periodic eval + save
            if i_so_far % self.save_freq == 0:
                # Decide whether to record video this evaluation
                record_video = (i_so_far % self.video_freq == 0)
                
                mean_ret, std_ret, frames, ep_returns = self.evaluate_policy(
                    self.eval_env, 
                    episodes=3,
                    record_video=record_video
                )
                
                # Save video if recorded
                if record_video and frames:
                    os.makedirs('videos', exist_ok=True)
                    video_path = f'videos/{self.run_name}_iter{i_so_far}.mp4'
                    save_video(frames, video_path)
                    try:
                        wandb.log({"eval/video": wandb.Video(video_path, fps=30, format="mp4")})
                    except Exception as e:
                        print(f"Failed to log video to wandb: {e}")
                
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

    # Rollout & Evaluation

    def rollout(self) -> int:
        """Collect timesteps_per_batch steps into the replay buffer."""
        count = 0
        obs, _ = self.env.reset()
        done = False
        ep_len = 0
        ep_rews = 0.0

        while count < self.timesteps_per_batch:
            # Random exploration for initial steps
            if self.logger['t_so_far'] < self.start_steps:
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

    def evaluate_policy(self, env, episodes: int = 5, max_episode_steps: Optional[int] = None,
                        render: bool = False, record_video: bool = False) -> Tuple[float, float, List, List]:
        """
        Evaluate policy deterministically with optional video recording.
        
        Args:
            env: Evaluation environment
            episodes: Number of episodes to run
            max_episode_steps: Max steps per episode (None = no limit)
            render: Whether to render to screen
            record_video: Whether to record frames for video
        
        Returns:
            mean_return: Average episode return
            std_return: Std of episode returns
            frames: List of frame lists (one per episode) if record_video=True, else empty
            returns: List of individual episode returns
        """
        returns = []
        all_frames = []

        for ep in range(episodes):
            s, _ = env.reset()
            done = False
            ep_ret = 0.0
            steps = 0
            ep_frames = []

            while not done:
                action, _ = self.get_action(s, deterministic=True)
                s, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_ret += reward
                steps += 1

                if render:
                    env.render()
                
                if record_video:
                    frame = env.render()
                    if frame is not None:
                        # Resize for smaller file size
                        frame_small = cv2.resize(frame, (320, 240))
                        ep_frames.append(frame_small)

                if max_episode_steps and steps >= max_episode_steps:
                    break

            returns.append(ep_ret)
            if record_video:
                all_frames.append(ep_frames)

        return float(np.mean(returns)), float(np.std(returns)), all_frames, returns

    # Action Selection

    def get_action(self, obs, deterministic: bool = False):
        """
        Get action for environment interaction.
        
        Args:
            obs: single observation (np array) or batch
            deterministic: if True, use mean action
            
        Returns:
            action_np: numpy array of action(s) in env scale [low, high]
            log_prob_np: numpy array of log probs or None if deterministic
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _ = self.actor(obs_tensor, deterministic=deterministic)

        # Scale action from [-1, 1] to [low, high]
        action_scaled = self._rescale_action(action)

        action_np = action_scaled.cpu().numpy()
        if action_np.shape[0] == 1:
            action_np = action_np[0]

        if log_prob is None:
            return action_np, None
        else:
            log_prob_np = log_prob.cpu().numpy()
            if log_prob_np.shape[0] == 1:
                log_prob_np = log_prob_np[0]
            return action_np, log_prob_np

    def _rescale_action(self, action_unscaled: torch.Tensor) -> torch.Tensor:
        """Map action from [-1, 1] to [low, high]."""
        return self.action_low + 0.5 * (action_unscaled + 1.0) * (self.action_high - self.action_low)

    def _soft_update(self, source_net: nn.Module, target_net: nn.Module):
        """Polyak averaging for target networks."""
        for param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    # Logging

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
        print(f"Alpha: {float(self.alpha.detach().cpu()):.4f}")
        print(f"Replay Buffer Size: {len(self.replay_buffer)}")
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
            "replay_buffer_size": len(self.replay_buffer),
            "iteration_duration_sec": float(delta_t),
        }, step=self.logger['i_so_far'])

        # Reset per-iteration buffers
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
        self.logger['alpha_losses'] = []












# """
# Refactored SAC implementation matching the working module_fsac/SAC architecture.
# Key changes from original sac_test.py:
# 1. Larger hidden layers (256x256 instead of 64x64)
# 2. State-dependent log_std (separate network head)
# 3. Proper weight initialization (orthogonal)
# 4. Proper Gaussian policy with mean/log_std heads
# """

# import time
# import random
# import math
# from collections import deque, namedtuple
# from typing import Deque, Tuple

# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch.distributions import Normal
# import wandb


# # Utility Functions & Classes

# Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


# def init_weights(m):
#     """Orthogonal initialization for linear layers."""
#     if isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0.0)


# class ReplayBuffer:
#     def __init__(self, capacity: int, device: torch.device):
#         self.capacity = capacity
#         self.device = device
#         self.buffer: Deque[Transition] = deque(maxlen=capacity)

#     def push(self, *args):
#         self.buffer.append(Transition(*args))

#     def sample(self, batch_size: int):
#         batch = random.sample(self.buffer, batch_size)
#         states = torch.tensor(
#             np.array([b.state for b in batch]),
#             dtype=torch.float32,
#             device=self.device,
#         )
#         actions = torch.tensor(
#             np.array([b.action for b in batch]),
#             dtype=torch.float32,
#             device=self.device,
#         )
#         rewards = torch.tensor(
#             np.array([b.reward for b in batch]),
#             dtype=torch.float32,
#             device=self.device,
#         ).unsqueeze(1)
#         next_states = torch.tensor(
#             np.array([b.next_state for b in batch]),
#             dtype=torch.float32,
#             device=self.device,
#         )
#         dones = torch.tensor(
#             np.array([b.done for b in batch]),
#             dtype=torch.float32,
#             device=self.device,
#         ).unsqueeze(1)
#         return states, actions, rewards, next_states, dones

#     def __len__(self):
#         return len(self.buffer)


# # Network Classes

# class MLP(nn.Module):
#     """Standard MLP with orthogonal initialization."""
#     def __init__(self, inp_dim: int, out_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
#         super().__init__()
#         layers = []
#         last = inp_dim
#         for h in hidden_sizes:
#             layers.append(nn.Linear(last, h))
#             layers.append(nn.ReLU())
#             last = h
#         layers.append(nn.Linear(last, out_dim))
#         self.model = nn.Sequential(*layers)
#         self.model.apply(init_weights)

#     def forward(self, x):
#         return self.model(x)


# class QNetwork(nn.Module):
#     """Q-network: Q(s, a) -> scalar value."""
#     def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
#         super().__init__()
#         self.q = MLP(state_dim + action_dim, 1, hidden_sizes)

#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=-1)
#         return self.q(x)


# class GaussianPolicy(nn.Module):
#     """
#     Gaussian policy with reparameterization and tanh squashing.
#     Returns action, log_prob, and mean (pre-tanh mean).
    
#     Key difference from original: state-dependent log_std via separate head.
#     """
#     def __init__(self, state_dim: int, action_dim: int, 
#                  hidden_sizes: Tuple[int, ...] = (256, 256),
#                  log_std_min: float = -20, log_std_max: float = 2):
#         super().__init__()
#         self.log_std_min = log_std_min
#         self.log_std_max = log_std_max
#         self.action_dim = action_dim

#         # Shared trunk
#         layers = []
#         last = state_dim
#         for h in hidden_sizes:
#             layers.append(nn.Linear(last, h))
#             layers.append(nn.ReLU())
#             last = h
#         self.net = nn.Sequential(*layers)
        
#         # Separate heads for mean and log_std
#         self.mean_linear = nn.Linear(last, action_dim)
#         self.log_std_linear = nn.Linear(last, action_dim)
        
#         # Apply orthogonal init to trunk
#         self.net.apply(init_weights)
        
#         # Small uniform init for output layers (helps stability)
#         nn.init.uniform_(self.mean_linear.weight, -3e-3, 3e-3)
#         nn.init.uniform_(self.mean_linear.bias, -3e-3, 3e-3)
#         nn.init.uniform_(self.log_std_linear.weight, -3e-3, 3e-3)
#         nn.init.uniform_(self.log_std_linear.bias, -3e-3, 3e-3)

#     def forward(self, state: torch.Tensor, deterministic: bool = False):
#         """
#         Args:
#             state: (B, state_dim) tensor
#             deterministic: if True, return tanh(mean) without sampling
            
#         Returns:
#             action: (B, action_dim) in [-1, 1]
#             log_prob: (B, 1) or None if deterministic
#             mean: (B, action_dim) pre-tanh mean
#         """
#         x = self.net(state)
#         mean = self.mean_linear(x)
#         log_std = self.log_std_linear(x)
#         log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
#         std = torch.exp(log_std)

#         if deterministic:
#             action = torch.tanh(mean)
#             return action, None, mean

#         # Reparameterization trick
#         normal = Normal(mean, std)
#         z = normal.rsample()
#         action = torch.tanh(z)

#         # Log prob with tanh squashing correction
#         # log π(a|s) = log π(z|s) - sum(log(1 - tanh²(z)))
#         log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)
#         # Numerically stable correction
#         eps = 1e-6
#         log_prob -= torch.log(1 - action.pow(2) + eps).sum(dim=-1, keepdim=True)
        
#         return action, log_prob, mean


# # SAC Agent

# class SAC_self:
#     """
#     Soft Actor-Critic implementation matching the PPO-style API.
    
#     Uses:
#       * Proper GaussianPolicy with state-dependent std
#       * Twin Q-networks + target networks (256x256 hidden)
#       * Automatic entropy tuning (alpha)
#       * Orthogonal weight initialization
#     """

#     def __init__(self, policy_class, env, eval_env, **hyperparameters):
#         """
#         Args:
#             policy_class: Ignored - we use our own GaussianPolicy and QNetwork
#             env: Training environment
#             eval_env: Evaluation environment
#             **hyperparameters: SAC hyperparameters
#         """
#         # Env checks
#         assert isinstance(env.observation_space, gym.spaces.Box)
#         assert isinstance(env.action_space, gym.spaces.Box)

#         self.env = env
#         self.eval_env = eval_env
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.obs_dim = env.observation_space.shape[0]
#         self.act_dim = env.action_space.shape[0]
#         self.action_low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device=self.device)
#         self.action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device=self.device)

#         # Initialize hyperparameters first
#         self._init_hyperparameters(hyperparameters)

#         # Networks with proper architecture
#         hidden_sizes = self.hidden_sizes
        
#         # Actor: Gaussian policy with state-dependent std
#         self.actor = GaussianPolicy(
#             self.obs_dim, self.act_dim, 
#             hidden_sizes=hidden_sizes
#         ).to(self.device)
        
#         # Critics: Q(s,a) networks + targets
#         self.critic1 = QNetwork(self.obs_dim, self.act_dim, hidden_sizes=hidden_sizes).to(self.device)
#         self.critic2 = QNetwork(self.obs_dim, self.act_dim, hidden_sizes=hidden_sizes).to(self.device)
#         self.critic1_target = QNetwork(self.obs_dim, self.act_dim, hidden_sizes=hidden_sizes).to(self.device)
#         self.critic2_target = QNetwork(self.obs_dim, self.act_dim, hidden_sizes=hidden_sizes).to(self.device)

#         # Copy weights to targets
#         self.critic1_target.load_state_dict(self.critic1.state_dict())
#         self.critic2_target.load_state_dict(self.critic2.state_dict())

#         # Optimizers
#         self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
#         self.critic1_optim = Adam(self.critic1.parameters(), lr=self.lr)
#         self.critic2_optim = Adam(self.critic2.parameters(), lr=self.lr)

#         # Automatic entropy tuning
#         if self.auto_entropy_tuning:
#             self.target_entropy = -float(self.act_dim)  # Heuristic: -|A|
#             self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
#             self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
#         else:
#             self.log_alpha = None
#             self._alpha_fixed = torch.tensor(self.alpha_fixed, device=self.device)
#             self.alpha_optim = None
#             self.target_entropy = None

#         # Replay buffer
#         self.replay_buffer = ReplayBuffer(self.replay_capacity, self.device)

#         # Logger
#         self.logger = {
#             'delta_t': time.time_ns(),
#             't_so_far': 0,
#             'i_so_far': 0,
#             'batch_lens': [],
#             'batch_rews': [],
#             'actor_losses': [],
#             'critic_losses': [],
#             'alpha_losses': [],
#         }

#     def _init_hyperparameters(self, hyperparameters):
#         """Initialize hyperparameters with sensible defaults."""
#         # Core SAC params
#         self.lr = hyperparameters.get('lr', 3e-4)
#         self.gamma = hyperparameters.get('gamma', 0.99)
#         self.tau = hyperparameters.get('tau', 0.005)

#         # Network architecture
#         self.hidden_sizes = hyperparameters.get('hidden_sizes', (256, 256))

#         # Data collection / updates
#         self.timesteps_per_batch = hyperparameters.get('timesteps_per_batch', 4096)
#         self.max_timesteps_per_episode = hyperparameters.get('max_timesteps_per_episode', 1000)
#         self.n_updates_per_iteration = hyperparameters.get('n_updates_per_iteration', 5)
#         self.batch_size = hyperparameters.get('batch_size', 256)
#         self.replay_capacity = hyperparameters.get('replay_capacity', 500_000)
#         self.start_steps = hyperparameters.get('start_steps', 5000)  # Random exploration steps

#         # Entropy
#         self.auto_entropy_tuning = hyperparameters.get('auto_entropy_tuning', True)
#         self.alpha_fixed = hyperparameters.get('alpha', 0.2)

#         # Logging / saving
#         self.save_freq = hyperparameters.get('save_freq', 10)
#         self.render = hyperparameters.get('render', False)
#         self.run_name = hyperparameters.get('run_name', 'sac_run')

#     @property
#     def alpha(self):
#         if self.log_alpha is None:
#             return self._alpha_fixed
#         else:
#             return self.log_alpha.exp()

#     # Training

#     def learn(self, total_timesteps: int):
#         print(f"Learning SAC... Running up to {total_timesteps} timesteps")
#         print(f"{self.timesteps_per_batch} timesteps per batch")
#         print(f"Using hidden sizes: {self.hidden_sizes}")

#         t_so_far = 0
#         i_so_far = 0

#         while t_so_far < total_timesteps:
#             collected = self.rollout()
#             t_so_far += collected
#             i_so_far += 1

#             self.logger['t_so_far'] = t_so_far
#             self.logger['i_so_far'] = i_so_far

#             # Skip updates if not enough data
#             if len(self.replay_buffer) < self.batch_size:
#                 self._log_summary()
#                 continue

#             actor_losses = []
#             critic_losses = []
#             alpha_losses = []

#             for _ in range(self.n_updates_per_iteration):
#                 states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

#                 # ============ Critic Update ============
#                 with torch.no_grad():
#                     # Sample next actions from current policy
#                     next_actions, next_log_probs, _ = self.actor(next_states)
                    
#                     # Compute target Q values
#                     q1_next = self.critic1_target(next_states, next_actions)
#                     q2_next = self.critic2_target(next_states, next_actions)
#                     q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_probs
#                     target_q = rewards + self.gamma * (1.0 - dones) * q_next

#                 # Current Q estimates
#                 q1 = self.critic1(states, actions)
#                 q2 = self.critic2(states, actions)

#                 # Critic losses
#                 critic1_loss = F.mse_loss(q1, target_q)
#                 critic2_loss = F.mse_loss(q2, target_q)
#                 critic_loss = critic1_loss + critic2_loss

#                 self.critic1_optim.zero_grad()
#                 self.critic2_optim.zero_grad()
#                 critic_loss.backward()
#                 self.critic1_optim.step()
#                 self.critic2_optim.step()

#                 # ============ Actor Update ============
#                 new_actions, log_probs, _ = self.actor(states)
#                 q1_pi = self.critic1(states, new_actions)
#                 q2_pi = self.critic2(states, new_actions)
#                 q_pi = torch.min(q1_pi, q2_pi)

#                 # Actor loss: maximize Q - alpha * log_prob
#                 actor_loss = (self.alpha.detach() * log_probs - q_pi).mean()

#                 self.actor_optim.zero_grad()
#                 actor_loss.backward()
#                 self.actor_optim.step()

#                 # ============ Alpha Update ============
#                 if self.auto_entropy_tuning:
#                     alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
#                     self.alpha_optim.zero_grad()
#                     alpha_loss.backward()
#                     self.alpha_optim.step()
#                 else:
#                     alpha_loss = torch.zeros(1, device=self.device)

#                 # ============ Soft Update Targets ============
#                 self._soft_update(self.critic1, self.critic1_target)
#                 self._soft_update(self.critic2, self.critic2_target)

#                 actor_losses.append(actor_loss.detach())
#                 critic_losses.append(critic_loss.detach())
#                 alpha_losses.append(alpha_loss.detach())

#             # Log losses
#             self.logger['actor_losses'].append(torch.stack(actor_losses).mean())
#             self.logger['critic_losses'].append(torch.stack(critic_losses).mean())
#             self.logger['alpha_losses'].append(torch.stack(alpha_losses).mean())

#             self._log_summary()

#             # Periodic eval + save
#             if i_so_far % self.save_freq == 0:
#                 mean_ret, std_ret = self.evaluate_policy(self.eval_env, episodes=3)
#                 wandb.log({
#                     "eval/mean": mean_ret,
#                     "eval/std": std_ret,
#                     "alpha": float(self.alpha.detach().cpu()),
#                 }, step=self.logger['i_so_far'])

#                 torch.save(self.actor.state_dict(), './sac_actor.pth')
#                 torch.save(self.critic1.state_dict(), './sac_critic1.pth')
#                 torch.save(self.critic2.state_dict(), './sac_critic2.pth')
#                 wandb.save(f"{self.run_name}_actor_iter{i_so_far}.pth")
#                 wandb.save(f"{self.run_name}_critic1_iter{i_so_far}.pth")
#                 wandb.save(f"{self.run_name}_critic2_iter{i_so_far}.pth")

#     # Rollout & Evaluation

#     def rollout(self) -> int:
#         """Collect timesteps_per_batch steps into the replay buffer."""
#         count = 0
#         obs, _ = self.env.reset()
#         done = False
#         ep_len = 0
#         ep_rews = 0.0

#         while count < self.timesteps_per_batch:
#             # Random exploration for initial steps
#             if self.logger['t_so_far'] < self.start_steps:
#                 action = self.env.action_space.sample()
#             else:
#                 action, _ = self.get_action(obs)

#             next_obs, reward, terminated, truncated, _ = self.env.step(action)
#             done = terminated or truncated

#             self.replay_buffer.push(obs, action, reward, next_obs, float(done))

#             count += 1
#             obs = next_obs
#             ep_len += 1
#             ep_rews += reward

#             if done or ep_len >= self.max_timesteps_per_episode:
#                 self.logger['batch_lens'].append(ep_len)
#                 self.logger['batch_rews'].append(ep_rews)

#                 obs, _ = self.env.reset()
#                 done = False
#                 ep_len = 0
#                 ep_rews = 0.0

#         return count

#     def evaluate_policy(self, env, episodes: int = 5, max_episode_steps=None, render=False):
#         """Evaluate policy deterministically."""
#         returns = []

#         for _ in range(episodes):
#             s, _ = env.reset()
#             done = False
#             ep_ret = 0.0
#             steps = 0

#             while not done:
#                 action, _ = self.get_action(s, deterministic=True)
#                 s, reward, terminated, truncated, _ = env.step(action)
#                 done = terminated or truncated
#                 ep_ret += reward
#                 steps += 1

#                 if max_episode_steps and steps >= max_episode_steps:
#                     break
#                 if render:
#                     env.render()

#             returns.append(ep_ret)

#         return float(np.mean(returns)), float(np.std(returns))

#     # Action Selection

#     def get_action(self, obs, deterministic: bool = False):
#         """
#         Get action for environment interaction.
        
#         Args:
#             obs: single observation (np array) or batch
#             deterministic: if True, use mean action
            
#         Returns:
#             action_np: numpy array of action(s) in env scale [low, high]
#             log_prob_np: numpy array of log probs or None if deterministic
#         """
#         obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
#         if obs_tensor.dim() == 1:
#             obs_tensor = obs_tensor.unsqueeze(0)

#         with torch.no_grad():
#             action, log_prob, _ = self.actor(obs_tensor, deterministic=deterministic)

#         # Scale action from [-1, 1] to [low, high]
#         action_scaled = self._rescale_action(action)

#         action_np = action_scaled.cpu().numpy()
#         if action_np.shape[0] == 1:
#             action_np = action_np[0]

#         if log_prob is None:
#             return action_np, None
#         else:
#             log_prob_np = log_prob.cpu().numpy()
#             if log_prob_np.shape[0] == 1:
#                 log_prob_np = log_prob_np[0]
#             return action_np, log_prob_np

#     def _rescale_action(self, action_unscaled: torch.Tensor) -> torch.Tensor:
#         """Map action from [-1, 1] to [low, high]."""
#         return self.action_low + 0.5 * (action_unscaled + 1.0) * (self.action_high - self.action_low)

#     def _soft_update(self, source_net: nn.Module, target_net: nn.Module):
#         """Polyak averaging for target networks."""
#         for param, target_param in zip(source_net.parameters(), target_net.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

#     # Logging

#     def _log_summary(self):
#         delta_t_prev = self.logger['delta_t']
#         self.logger['delta_t'] = time.time_ns()
#         delta_t = (self.logger['delta_t'] - delta_t_prev) / 1e9

#         avg_ep_lens = np.mean(self.logger['batch_lens']) if self.logger['batch_lens'] else 0.0
#         avg_ep_rews = np.mean(self.logger['batch_rews']) if self.logger['batch_rews'] else 0.0

#         def _mean_loss(key):
#             if not self.logger[key]:
#                 return 0.0
#             vals = [x.float().cpu().item() for x in self.logger[key]]
#             return float(np.mean(vals))

#         avg_actor_loss = _mean_loss('actor_losses')
#         avg_critic_loss = _mean_loss('critic_losses')
#         avg_alpha_loss = _mean_loss('alpha_losses')

#         print(flush=True)
#         print(f"-------------------- Iteration #{self.logger['i_so_far']} --------------------")
#         print(f"Average Episodic Length: {avg_ep_lens:.2f}")
#         print(f"Average Episodic Return: {avg_ep_rews:.2f}")
#         print(f"Average Actor Loss: {avg_actor_loss:.5f}")
#         print(f"Average Critic Loss: {avg_critic_loss:.5f}")
#         print(f"Average Alpha Loss: {avg_alpha_loss:.5f}")
#         print(f"Alpha: {float(self.alpha.detach().cpu()):.4f}")
#         print(f"Replay Buffer Size: {len(self.replay_buffer)}")
#         print(f"Iteration took: {delta_t:.2f} secs")
#         print("------------------------------------------------------", flush=True)

#         wandb.log({
#             "iteration": self.logger['i_so_far'],
#             "timesteps_so_far": self.logger['t_so_far'],
#             "avg_episode_length": float(avg_ep_lens),
#             "avg_episode_return": float(avg_ep_rews),
#             "avg_actor_loss": float(avg_actor_loss),
#             "avg_critic_loss": float(avg_critic_loss),
#             "avg_alpha_loss": float(avg_alpha_loss),
#             "alpha": float(self.alpha.detach().cpu()),
#             "replay_buffer_size": len(self.replay_buffer),
#             "iteration_duration_sec": float(delta_t),
#         }, step=self.logger['i_so_far'])

#         # Reset per-iteration buffers
#         self.logger['batch_lens'] = []
#         self.logger['batch_rews'] = []
#         self.logger['actor_losses'] = []
#         self.logger['critic_losses'] = []
#         self.logger['alpha_losses'] = []