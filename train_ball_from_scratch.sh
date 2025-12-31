#!/bin/bash
# 带球训练脚本 - 从零开始训练

# WandB配置
export WANDB_BASE_URL=https://api.bandw.top

# 代理设置
export https_proxy=http://100.68.175.95:3128
export http_proxy=http://100.68.175.95:3128

# 训练参数
echo "================================"
echo "带球训练 - 从零开始"
echo "================================"

# 基础训练命令
python launch.py \
    task=HumanoidAMPBallTarget \
    train=HumanoidAMPBallTargetPPO \
    headless=True \
    wandb_activate=True

