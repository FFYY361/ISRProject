#!/bin/bash
# 带球训练脚本 - 从已有checkpoint继续训练

# WandB配置
export WANDB_BASE_URL=https://api.bandw.top

# 代理设置
# export https_proxy=http://100.68.175.95:3128
# export http_proxy=http://100.68.175.95:3128

# 检查是否提供了checkpoint路径
if [ -z "$1" ]; then
    echo "错误: 请提供checkpoint路径"
    echo "用法: $0 <checkpoint_path>"
    echo "示例: $0 ./runs/HumanoidAMP_25-08-17-47/nn/last_HumanoidAMP.pth"
    exit 1
fi

CHECKPOINT_PATH=$1

# 检查文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: checkpoint文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "================================"
echo "带球训练 - 从checkpoint继续"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "================================"


python launch.py \
    task=HumanoidAMPBall \
    train=HumanoidAMPBallPPO \
    headless=True \
    checkpoint="$CHECKPOINT_PATH" \
    num_envs=4096 \
    wandb_activate=True

