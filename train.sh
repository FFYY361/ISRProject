#!/bin/bash
export WANDB_BASE_URL=https://api.bandw.top

python launch.py \
    task=HumanoidAMP \
    train=HumanoidAMPPPO \
    headless=True \
    wandb_activate=True \
    num_envs=4096 \

