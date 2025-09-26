# FPO Training Environment

This repository provides a reproducible training environment and scripts for running **Flow-based Policy Optimization (FPO)** experiments.  

## Setup

First, create and activate the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate flow_mujoco   # or the name defined in environment.yml
```
## 🚀 Training Scripts

Two training scripts are available:

1. **Gridworld (FPO method)**

```bash
python fpo/gridworld/main.py --method fpo
```
2. **Playground with W&B logging**

```bash
python fpo/playground/scripts/train_fpo.py \
    --wandb-entity hl3811-columbia-university \
    --wandb-project sac-pendulum \
    --env-name AcrobotSwingup
```