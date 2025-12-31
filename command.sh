export WANDB_BASE_URL=https://api.bandw.top
export https_proxy=http://100.68.175.95:3128
export http_proxy=http://100.68.175.95:3128


# runs/HumanoidAMP_23-21-59-30/nn
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python launch.py task=HumanoidAMPBallTarget headless=True test=True num_envs=1 checkpoint=/root/code/ISRProject/runs/HumanoidAMPBall_31-05-56-08/nn/HumanoidAMPBall_31-05-56-24_4500.pth capture_video=True


# export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# export NVIDIA_DRIVER_CAPABILITIES=all
# # 关键：告诉渲染器在没有显示器的情况下使用 EGL
# export PYOPENGL_PLATFORM=egl

# ls /lib/x86_64-linux-gnu/libnvidia-vulkan*
# find /usr/lib/x86_64-linux-gnu/ -name "libGLX_nvidia.so*"
# vulkaninfo | grep -i nvidia
