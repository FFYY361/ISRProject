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

python launch.py \
    task=HumanoidAMPBallTarget \
    headless=True \
    test=True \
    num_envs=1 \
    checkpoint=$CHECKPOINT_PATH \
    capture_video=True \


