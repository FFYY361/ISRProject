## Paper Reproduction: AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control

### Environment Setup

Ensure your system environment is Linux with Python 3.8, and that you have a display connected (or use a virtual display with 3D acceleration) for rendering.

First, download the Isaac Gym Preview 4 physics simulation environment [here](https://developer.nvidia.com/isaac-gym), follow the installation instructions, and verify that the example programs run correctly: `python/examples`, such as `joint_monkey.py`.

Then, install the required packages:

```
"gym==0.23.1",
"torch",
"omegaconf",
"termcolor",
"jinja2",
"hydra-core>=1.2",
"rl-games>=1.6.0",
"pyvirtualdisplay",
"urdfpy==0.0.22",
"pysdf==0.1.9",
"warp-lang==0.10.1",
"trimesh==3.23.5",
```

### Training Models

This project provides three training scripts for different scenarios:

#### 1. Basic HumanoidAMP Training (`train.sh`)

This script trains a basic humanoid character using the AMP (Adversarial Motion Priors) framework without any ball interaction. It uses:
- **Task Config**: `HumanoidAMP.yaml` - Basic humanoid locomotion
- **Train Config**: `HumanoidAMPPPO.yaml` - Standard PPO training with AMP discriminator
- **Motion File**: Choose from provided dataset

To train:
```bash
bash train.sh
```

Or manually:
```bash
python launch.py \
    task=HumanoidAMP \
    train=HumanoidAMPPPO \
    headless=True \
    wandb_activate=True \
    num_envs=4096
```

#### 2. Ball Dribbling with Target - From Scratch (`train_ball_from_scratch.sh`)

This script trains a humanoid to dribble a ball toward a target position from scratch. It uses:
- **Task Config**: `HumanoidAMPBallTarget.yaml` - Ball interaction with target or `HumanoidAMPBallNoTarget.yaml` - Ball interaction without target
  - `motion_file: "amp_humanoid_run.npy"` - Uses running motion
  - `task_speed: 2.5` - Target forward speed
- **Train Config**: `HumanoidAMPBallTargetPPO.yaml` - PPO training optimized for dribbling target task or `HumanoidAMPBallTargetPPO.yaml` - PPO training optimized for dribbling forward task
  - `reward_combine: 'add'` - Task reward and discriminator reward are added
  - `task_reward_w: 1.0`, `disc_reward_w: 0.2` - Reward weights

To train:
```bash
bash train_ball_from_scratch.sh
```

Or manually:
```bash
python launch.py \
    task=HumanoidAMPBallTarget \
    train=HumanoidAMPBallTargetPPO \
    headless=True \
    wandb_activate=True \
    num_envs=4096
```

#### 3. Ball Dribbling without Target - From Checkpoint (`train_ball_from_checkpoint.sh`)

This script continues training a humanoid to dribble a ball without a target position, starting from a pre-trained checkpoint. It uses:
- **Task Config**: `HumanoidAMPBallTarget.yaml` - Ball interaction with target or `HumanoidAMPBallNoTarget.yaml` - Ball interaction without target
  - `motion_file: "amp_humanoid_run.npy"` - Uses running motion
  - `task_speed: 2.5` - Target forward speed
- **Train Config**: `HumanoidAMPBallTargetPPO.yaml` - PPO training optimized for dribbling target task or `HumanoidAMPBallTargetPPO.yaml` - PPO training optimized for dribbling forward task 
  - `reward_combine: 'add'` - Task reward and discriminator reward are added
  - `task_reward_w: 1.0`, `disc_reward_w: 0.2` - Reward weights

To train:
```bash
bash train_ball_from_checkpoint.sh <checkpoint_path>
```

Example:
```bash
bash train_ball_from_checkpoint.sh ./runs/HumanoidAMPBall_31-08-07-42/nn/HumanoidAMPBall_31-08-08-00.pth
```

Or manually:
```bash
python launch.py \
    task=HumanoidAMPBall \
    train=HumanoidAMPBallPPO \
    headless=True \
    checkpoint="<checkpoint_path>" \
    num_envs=4096 \
    wandb_activate=True
```


### Test Trained Models

After training, test your model with:

```bash
python launch.py \
    task=<TaskName> \
    headless=False \
    test=True \
    num_envs=64 \
    checkpoint=/path/to/saved/model/in/runs/nn/<model_name>.pth
```

### Rendering Results to Video

To render trained model results to video, use the following command:

```bash
python launch.py \
    task=<TaskName> \
    headless=False \
    test=True \
    num_envs=64 \
    checkpoint=/path/to/saved/model/in/runs/nn/<model_name>.pth \
    capture_video=True
```

Replace `<TaskName>` with the appropriate task name (e.g., `HumanoidAMP`, `HumanoidAMPBallTarget`, or `HumanoidAMPBall`).

The rendered videos will be saved in the current working directory.
