#!/bin/bash
export WANDB_BASE_URL=https://api.bandw.top
# 代理设置
# export https_proxy=http://100.68.175.95:3128
# export http_proxy=http://100.68.175.95:3128

# ============================================
# Walk训练参数配置（第一阶段：不带球）
# ============================================

# 基础参数（必须）
TASK="HumanoidAMPWalk"
TRAIN_CONFIG="HumanoidAMPWalkPPO"
HEADLESS="True"

# 可选参数（根据需要修改）
# CHECKPOINT=""  # 第一阶段从零开始，不需要checkpoint
# NUM_ENVS=""  # 环境数量，默认4096，如果GPU内存不足可以减少，例如: "2048"
# MAX_ITERATIONS=""  # 最大训练轮数，例如: "2000"
# SEED=""  # 随机种子，例如: "42"

# ============================================
# 构建训练命令
# ============================================

TRAIN_CMD="python launch.py task=$TASK train=$TRAIN_CONFIG headless=$HEADLESS num_envs=4096 wandb_activate=True"

# 添加可选参数
if [ ! -z "$CHECKPOINT" ]; then
    TRAIN_CMD="$TRAIN_CMD checkpoint=$CHECKPOINT"
fi

if [ ! -z "$NUM_ENVS" ]; then
    TRAIN_CMD="$TRAIN_CMD num_envs=$NUM_ENVS"
fi

if [ ! -z "$MAX_ITERATIONS" ]; then
    TRAIN_CMD="$TRAIN_CMD max_iterations=$MAX_ITERATIONS"
fi

if [ ! -z "$SEED" ]; then
    TRAIN_CMD="$TRAIN_CMD seed=$SEED"
fi

# ============================================
# 执行训练
# ============================================

echo "================================"
echo "Walk训练启动（第一阶段：不带球）"
echo "================================"
echo "任务: $TASK"
echo "训练配置: $TRAIN_CONFIG"
echo "无头模式: $HEADLESS"
echo "注意: 此配置不加载球体，用于学习基础walk技能"
echo "训练完成后，可以使用train_ball_from_checkpoint.sh继续训练带球版本"
[ ! -z "$CHECKPOINT" ] && echo "Checkpoint: $CHECKPOINT"
[ ! -z "$NUM_ENVS" ] && echo "环境数量: $NUM_ENVS"
[ ! -z "$MAX_ITERATIONS" ] && echo "最大轮数: $MAX_ITERATIONS"
echo "================================"
echo "训练命令:"
echo "$TRAIN_CMD"
echo "================================"

$TRAIN_CMD

