import jax
import jax.numpy as jnp
from daxbench.core.envs import ShapeRopeEnv

# Crreate the environments
env = ShapeRopeEnv(batch_size=3, seed=1)
obs, state = env.reset(env.simulator.key)

# Actions to be simulated in each environment
actions = jnp.array(
    [
        [0.4, 0, 0.4, 0.6, 0, 0.6],
        [0.6, 0, 0.6, 0.4, 0, 0.4],
        [0.4, 0, 0.6, 0.6, 0, 0.4],
    ]
)

# without jax
# obs, reward, done, info = env.step_with_render(actions, state)
# next_state = info["state"]


# with jax
obs, reward, done, info = env.step_diff(actions, state)
next_state = info["state"]
image = env.render(next_state, visualize=True)
# print(next_state)