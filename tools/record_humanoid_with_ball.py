import numpy as np
import imageio
from isaacgym import gymapi, gymutil

def main():
    # 1. 初始化 Gym 和 参数
    gym = gymapi.acquire_gym()
    
    # 核心：确保在服务器运行时添加 --headless 参数
    args = gymutil.parse_arguments(
        description="AMP Humanoid Dribbling - Record",
        custom_parameters=[]
    )

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0, 0, -9.81)

    # 使用 Flex 或 PhysX (AMP 通常建议使用 PhysX)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.0

    sim = gym.create_sim(
        args.compute_device_id, 
        args.graphics_device_id, 
        args.physics_engine, 
        sim_params
    )

    if sim is None:
        print("*** Failed to create sim")
        quit()

    # 2. 地面
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    # 3. 加载资产 (Asset)
    asset_root = "/root/code/ISRProject/assets"
    humanoid_file = "mjcf/amp_humanoid.xml"
    ball_file = "urdf/objects/ball.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.5
    asset_options.max_angular_velocity = 100.0
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    
    humanoid_asset = gym.load_asset(sim, asset_root, humanoid_file, asset_options)
    
    # 球体资产配置：增加弹性以便踢球
    ball_options = gymapi.AssetOptions()
    ball_options.fill_inertia_from_mesh = True
    ball_options.density = 100.0 # 调整重量，避免太轻被踢飞
    ball_asset = gym.load_asset(sim, asset_root, ball_file, ball_options)

    # 4. 创建环境
    spacing = 2.0
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, lower, upper, 1)

    # 放置 Humanoid
    start_pose = gymapi.Transform()
    start_pose.p = gymapi.Vec3(0.0, 0.0, 1.32) # AMP 人型通常需要略高于地面
    humanoid_handle = gym.create_actor(env, humanoid_asset, start_pose, "humanoid", 0, 1)

    # 放置球 (就在人脚边)
    ball_pose = gymapi.Transform()
    ball_pose.p = gymapi.Vec3(0.6, 0.0, 0.2) 
    ball_handle = gym.create_actor(env, ball_asset, ball_pose, "ball", 0, 2)

    # 5. DOF 属性设置
    dof_props = gym.get_actor_dof_properties(env, humanoid_handle)
    dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
    dof_props['stiffness'].fill(300.0)
    dof_props['damping'].fill(30.0)
    gym.set_actor_dof_properties(env, humanoid_handle, dof_props)

    # 6. 相机设置
    cam_props = gymapi.CameraProperties()
    cam_props.width = 1280
    cam_props.height = 720
    cam_handle = gym.create_camera_sensor(env, cam_props)
    
    # 相机跟随位置
    cam_pos = gymapi.Vec3(3.0, 2.0, 2.0)
    cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)

    # 7. 仿真循环
    video_frames = []
    steps = 300 # 录制5秒 (60fps)
    
    print("Starting simulation and recording...")
    for i in range(steps):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # 重要：同步图形数据到 GPU
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

        # 获取图像
        raw_image = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
        
        # 转换数据格式
        img = raw_image.reshape(cam_props.height, cam_props.width, 4)
        video_frames.append(img[:, :, :3]) # 去掉 Alpha 通道

    # 8. 保存视频
    imageio.mimsave("dribbling_test.mp4", video_frames, fps=60)
    print("Saved to dribbling_test.mp4")

    gym.destroy_sim(sim)

if __name__ == '__main__':
    main()