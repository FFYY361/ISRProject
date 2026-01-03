#!/bin/bash
export WANDB_BASE_URL=https://api.bandw.top


if [ -z "$1" ]; then
    echo "Error: please provide checkpoint path"
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 ./runs/HumanoidAMPBall_31-08-07-42/nn/HumanoidAMPBall_31-08-08-00.pth"
    exit 1
fi

CHECKPOINT_PATH=$1

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: checkpoint file does not exist: $CHECKPOINT_PATH"
    exit 1
fi

echo "================================"
echo "Training Dribbling from checkpoint"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "================================"


python launch.py \
    task=HumanoidAMPBall \
    train=HumanoidAMPBallPPO \
    headless=True \
    checkpoint="$CHECKPOINT_PATH" \
    num_envs=4096 \
    wandb_activate=True

