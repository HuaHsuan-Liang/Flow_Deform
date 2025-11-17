import gymnasium as gym
import numpy as np

class NormalizeObsActWrapper(gym.Wrapper):
    """
    Normalize observations only if bounds are finite.
    Normalize actions from [-1,1] to env action space.
    """

    def __init__(self, env):
        super().__init__(env)

        # ------------ Action space normalization ------------
        assert isinstance(env.action_space, gym.spaces.Box)
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32
        )

        # ------------ Observation: check if normalizable ------------
        assert isinstance(env.observation_space, gym.spaces.Box)

        obs_low = env.observation_space.low
        obs_high = env.observation_space.high

        # Is normalization possible?
        self.use_obs_norm = np.isfinite(obs_low).all() and np.isfinite(obs_high).all()

        if self.use_obs_norm:
            # safe to normalize
            self.obs_low = obs_low
            self.obs_high = obs_high

            self.observation_space = gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=env.observation_space.shape,
                dtype=np.float32
            )
        else:
            # not normalizable — use original space unchanged
            self.observation_space = env.observation_space
            print("[NormalizeObsActWrapper] WARNING: Obs bounds contain inf → skipping observation normalization.")

    # ------------ normalization utils ------------

    def normalize_obs(self, obs):
        if not self.use_obs_norm:
            return obs  # no normalization
        return 2.0 * (obs - self.obs_low) / (self.obs_high - self.obs_low) - 1.0

    def denormalize_action(self, action):
        return self.act_low + (0.5 * (action + 1.0) * (self.act_high - self.act_low))

    # ------------ override ------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.normalize_obs(obs)
        return obs.astype(np.float32), info

    def step(self, action):
        action = self.denormalize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.normalize_obs(obs)
        return obs.astype(np.float32), reward, terminated, truncated, info
