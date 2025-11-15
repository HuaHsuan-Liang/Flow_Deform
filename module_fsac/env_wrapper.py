import gymnasium as gym
import numpy as np

class NormalizeObsActWrapper(gym.Wrapper):
    """
    Normalize observations to [-1, 1] (based on observation_space bounds)
    Normalize actions from [-1,1] (agent output) to env action space when calling step()

    Usage:
        env = NormalizeObsActWrapper(gym.make("HalfCheetah-v4"))
    """

    def __init__(self, env):
        super().__init__(env)

        # === Action space scaling ===
        assert isinstance(env.action_space, gym.spaces.Box), \
            "Action normalization only supports continuous (Box) action spaces."

        self.act_low = env.action_space.low
        self.act_high = env.action_space.high

        # === Observation normalization ===
        assert isinstance(env.observation_space, gym.spaces.Box), \
            "Obs normalization assumes Box obs space. For dict obs, wrap your encoder first."
        
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high

        # Replace observation space so the agent sees [-1, 1]
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

        # Replace action space so the agent outputs [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float32
        )

    # ---------- normalization utils ------------

    def normalize_obs(self, obs):
        """map obs from [low, high] to [-1, 1]"""
        return 2.0 * (obs - self.obs_low) / (self.obs_high - self.obs_low) - 1.0

    def denormalize_action(self, action):
        """map action from [-1, 1] to [low, high]"""
        return self.act_low + (0.5 * (action + 1.0) * (self.act_high - self.act_low))

    # ---------- override Gymnasium methods ------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalize_obs(obs).astype(np.float32), info

    def step(self, action):
        # agent outputs [-1,1], convert to real env action
        action = self.denormalize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.normalize_obs(obs).astype(np.float32), reward, terminated, truncated, info
