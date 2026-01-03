#!/bin/bash
export WANDB_BASE_URL=https://api.bandw.top

echo "================================"
echo "Training Dribbling from scratch"
echo "================================"

python launch.py \
    task=HumanoidAMPBallTarget \
    train=HumanoidAMPBallTargetPPO \
    headless=True \
    wandb_activate=True \
    num_envs=4096 \

