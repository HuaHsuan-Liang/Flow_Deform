from __future__ import annotations
from turtle import done

import gymnasium as gym
import numpy as np
import cv2
import numpy as np

from .SAC.agent import SACAgent
from .FSAC.agent import FSACAgent
# import gym
import time

def evaluate_policy(env, agent: SACAgent | FSACAgent, episodes=5, max_episode_steps=None,  record_video=False, video_fps=30):
    """
    Evaluate a policy and optionally log a video to wandb.

    Args:
        env: gym-like environment
        agent: policy agent (SACAgent or FSACAgent)
        episodes: number of evaluation episodes
        max_episode_steps: max steps per episode
        render: if True, call env.render() to show window
        record_video: if True, capture frames and log as wandb.Video
        video_fps: frames per second for the saved video
    """
    returns = []
    frames = []  # store RGB frames
    for _ in range(episodes):
        s, info = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        frames_ep = []
        while not done:
            a = agent.select_action(s, deterministic=True)
            s, r, terminated, truncated, info = env.step(a)
            # obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += r
            steps += 1
            
            if record_video:
                frame = env.render()
                frame_small = cv2.resize(frame, (320, 240))  # (width, height)
                frames_ep.append(frame_small)
            if max_episode_steps and steps >= max_episode_steps:
                break
        frames.append(frames_ep)
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns)), frames, returns