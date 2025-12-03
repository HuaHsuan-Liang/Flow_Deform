"""
        This file contains the arguments to parse at command line.
        File main.py will call get_args, which then the arguments
        will be returned.
"""
import argparse

def get_args():
        """
                Description:
                Parses arguments at command line.

                Parameters:
                        None

                Return:
                        args - the arguments parsed
        """
        # Allow extra arguments so wandb sweeps can pass hyperparameters
        # like `--lr`, `--timesteps_per_batch`, etc. without argparse failing.
        parser = argparse.ArgumentParser()

        parser.add_argument('--mode', dest='mode', type=str, default='train')              # can be 'train' or 'test'
        parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     # your actor model filename
        parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   # your critic model filename
        parser.add_argument('--method', dest='method', type=str, default='fpo')             # can be 'ppo', 'fpo', 'sac', 'sac_self', 'fpo_pg', or 'fpo_pg_gym'

        # Extra options for JAX FPO playground integration (method=fpo_pg)
        parser.add_argument('--pg_env_name', dest='pg_env_name', type=str, default='cartpole_swingup')
        parser.add_argument('--wandb_entity', dest='wandb_entity', type=str, default='')
        parser.add_argument('--wandb_project', dest='wandb_project', type=str, default='fpo-playground')
        
        # Training limits override
        parser.add_argument('--num_timesteps', dest='num_timesteps', type=int, default=60_000_000)
        parser.add_argument('--num_evals', dest='num_evals', type=int, default=10)

        # Use parse_known_args so unexpected flags are ignored (handled by wandb/config instead)
        args, _ = parser.parse_known_args()

        return args
