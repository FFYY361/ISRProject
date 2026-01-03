# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import cv2
import numpy as np
import os
import torch
from torch import Tensor
from typing import Tuple, Optional

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, get_axis_params, calc_heading_quat_inv, \
     exp_map_to_quat, quat_to_tan_norm, my_quat_rotate, calc_heading_quat_inv

from ..base.vec_task import VecTask

DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]

# Observation size
# Original: 13 (root) + 52 (dof_pos) + 28 (dof_vel) + 12 (key_body_pos) = 105
# Added ball info: 
#   Ball state: pos(3) + rot(4) + lin_vel(3) + ang_vel(3) = 13
#   Target: rel_pos(3) = 3
# Total added: 16

NUM_OBS = 13 + 52 + 28 + 12 + 13 + 3 
NUM_ACTIONS = 28


KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]

class HumanoidAMPBase(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):


        self.cfg = config

        self.task_speed = self.cfg["env"]["task_speed"]
        self.task_speed_mul = self.cfg["env"]["task_speed_mul"]

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()
        if virtual_screen_capture:
            self.cfg["env"]["enableCameraSensors"] = True

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt
        
        # get gym GPU state tensors
        self._actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        all_root_states = gymtorch.wrap_tensor(self._actor_root_state_tensor)

        if self._has_ball_asset:
            num_actors_per_env = 2  # humanoid + ball
            # Extract state of the first actor (humanoid) in each environment
            humanoid_indices = torch.arange(0, len(all_root_states), num_actors_per_env, device=self.device)
            self._root_states = all_root_states[humanoid_indices]
            # Extract state of the second actor (ball) in each environment
            ball_indices = torch.arange(1, len(all_root_states), num_actors_per_env, device=self.device)
            self._ball_root_states = all_root_states[ball_indices]
        else:
            self._root_states = all_root_states
            # Create zero tensor placeholder (for torch.jit.script compatibility)
            self._ball_root_states = torch.zeros((self.num_envs, 13), device=self.device, dtype=torch.float32)
        
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        right_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "right_shoulder_x")
        left_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "left_shoulder_x")
        self._initial_dof_pos[:, right_shoulder_x_handle] = 0.5 * np.pi
        self._initial_dof_pos[:, left_shoulder_x_handle] = -0.5 * np.pi

        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies, 3)
        
        self._target_states = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        if self.viewer != None:
            self._init_camera()
        if self.viewer is None and virtual_screen_capture:     
            self._init_headless_cameras()



        return

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        
        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._reset_env_tensors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        return

    def _reset_env_tensors(self, env_ids):
        n = len(env_ids)
        if n > 0:
            dist = 15.0 + 5.0 * torch.rand(n, device=self.device)
            angle = (2 * torch.rand(n, device=self.device) - 1.0) * np.pi / 2

            root_pos = self._root_states[env_ids, 0:3]
            root_rot = self._root_states[env_ids, 3:7]
            
            heading_rot = calc_heading_quat_inv(root_rot)
            target_local_pos = torch.zeros((n, 3), device=self.device)
            target_local_pos[:, 0] = dist * torch.cos(angle)
            target_local_pos[:, 1] = dist * torch.sin(angle)
            
            heading_rot_inv = calc_heading_quat_inv(root_rot)
            heading_rot = heading_rot_inv.clone()
            heading_rot[:, 0:3] = -heading_rot[:, 0:3]
            
            target_global_offset = my_quat_rotate(heading_rot, target_local_pos)
            self._target_states[env_ids, :3] = root_pos + target_global_offset
            
        return

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self.humanoid_handles[i]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        asset_file = "mjcf/amp_humanoid.xml"

        if "asset" in self.cfg["env"]:
            #asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Decide whether to load ball asset based on config (for ball environment)
        # Default is not to load ball, only load when explicitly specified in config
        self._has_ball_asset = self.cfg["env"].get("enableBall", False)
        self._enable_ball_reward = self.cfg["env"].get("enableBallReward", self._has_ball_asset)
        self._enable_target = self.cfg["env"].get("enableTarget", False)
        ball_asset_count = 0
        
        if self._has_ball_asset:
        # Use standard size soccer ball (radius 0.11m, approx 22cm diameter)
            ball_asset_file = "urdf/objects/soccer_ball.urdf"
            ball_asset_options = gymapi.AssetOptions()
            # Allow free movement
            ball_asset_options.fix_base_link = False
            # Set appropriate physical properties
            ball_asset_options.density = 100.0  # Density (kg/m^3), used to calculate mass
            ball_asset_options.override_com = False
            ball_asset_options.override_inertia = False

            ball_asset = self.gym.load_asset(self.sim, asset_root, ball_asset_file, ball_asset_options)
            ball_asset_count = self.gym.get_asset_rigid_body_count(ball_asset)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        # Number of rigid bodies per env: Humanoid + Optional Ball
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset) + (ball_asset_count if self._has_ball_asset else 0)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.ball_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            contact_filter = 0
            
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, contact_filter, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            # If there is a ball asset, create a ball actor for each env, placed in front of the character
            if self._has_ball_asset:
                ball_pose = gymapi.Transform()
                # Generate ball in front of humanoid start position (relative offset)
                # Ball placed approx 0.8m in front, height similar to feet (considering ball radius 0.11m)
                ball_pose.p = gymapi.Vec3(start_pose.p.x + 0.8, start_pose.p.y, 0.11)
                ball_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_pose, "ball", i, contact_filter, 0)
                self.ball_handles.append(ball_handle)
            else:
                self.ball_handles.append(-1)

            if (self._pd_control):
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)
        if self._has_ball_asset:
            # Use ball actor handle from 0th env to query body id (same in every env)
            self._ball_body_ids = self._build_ball_body_ids_tensor(self.envs[0], self.ball_handles[0])
        else:
            self._ball_body_ids = to_torch([-1], device=self.device, dtype=torch.long)
        
        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(DOF_OFFSETS) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = DOF_OFFSETS[j]
            dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

            if (dof_size == 3):
                lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reward(self, actions):
        # Get ball state and foot contact info
        if self._has_ball_asset:
            ball_root_states = self._ball_root_states
            # Extract foot contact forces from contact_forces (right_foot and left_foot)
            right_foot_id = self._key_body_ids[2]  # right_foot
            left_foot_id = self._key_body_ids[3]   # left_foot
            foot_contact_forces = self._contact_forces[:, [right_foot_id, left_foot_id], :]
        else:
            # Create zero tensor placeholder
            num_envs = self._root_states.shape[0]
            ball_root_states = torch.zeros((num_envs, 13), device=self.device, dtype=torch.float32)
            foot_contact_forces = torch.zeros((num_envs, 2, 3), device=self.device, dtype=torch.float32)
        
        self.rew_buf[:] = compute_humanoid_reward(
            self.obs_buf, 
            self.task_speed, 
            self.task_speed_mul,
            self._root_states,
            ball_root_states,
            foot_contact_forces,
            self._target_states,
            self._enable_ball_reward,
            self._enable_target
        )
        return

    def _compute_reset(self):
        contact_body_ids = self._contact_body_ids
        ball_root_states = None
        target_states = None
        
        if self._has_ball_asset:
            contact_body_ids = torch.cat((contact_body_ids, self._ball_body_ids), dim=-1)
            ball_root_states = self._ball_root_states
            target_states = self._target_states
            
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_height,
                                                   ball_root_states, target_states)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        # If there is a ball, need to extract humanoid and ball states from all actors' root states
        if self._has_ball_asset:
            all_root_states = gymtorch.wrap_tensor(self._actor_root_state_tensor)
            num_actors_per_env = 2  # humanoid + ball
            humanoid_indices = torch.arange(0, len(all_root_states), num_actors_per_env, device=self.device)
            self._root_states[:] = all_root_states[humanoid_indices]
            ball_indices = torch.arange(1, len(all_root_states), num_actors_per_env, device=self.device)
            self._ball_root_states[:] = all_root_states[ball_indices]
        
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            if self._has_ball_asset:
                ball_root_states = self._ball_root_states
            else:
                # Create zero tensor placeholder
                num_envs = root_states.shape[0]
                ball_root_states = torch.zeros((num_envs, 13), device=root_states.device, dtype=root_states.dtype)
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
            if self._has_ball_asset:
                ball_root_states = self._ball_root_states[env_ids]
            else:
                # Create zero tensor placeholder
                num_envs = len(env_ids)
                ball_root_states = torch.zeros((num_envs, 13), device=root_states.device, dtype=root_states.dtype)
        
        obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs, ball_root_states,
                                            self._target_states[env_ids] if env_ids is not None else self._target_states)
        return obs

    def _reset_actors(self, env_ids):
        if len(env_ids) == 0:
            return
        
        # Ensure env_ids are on the correct device and converted to long type for indexing
        if env_ids.device != self.device:
            env_ids = env_ids.to(self.device)
        env_ids = env_ids.long()  # Ensure it is long type for indexing
        
        # Ensure indices are within valid range and filter negative values
        env_ids = env_ids[(env_ids >= 0) & (env_ids < self.num_envs)]
        if len(env_ids) == 0:
            return
        
        # Ensure tensor is contiguous (avoid CUDA illegal memory access)
        env_ids = env_ids.contiguous()
        
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        
        # Update humanoid state in _actor_root_state_tensor
        if self._has_ball_asset:
            # If there is a ball, need to update all actor states
            # actor root state tensor structure: env0_actor0, env0_actor1, env1_actor0, env1_actor1, ...
            num_actors_per_env = 2
            # Get humanoid actor indices
            humanoid_actor_indices = env_ids * num_actors_per_env
            # Update humanoid state in _actor_root_state_tensor
            all_root_states = gymtorch.wrap_tensor(self._actor_root_state_tensor)
            all_root_states[humanoid_actor_indices] = self._initial_root_states[env_ids]
            
            # Reset ball state (position in front of humanoid, zero velocity)
            ball_actor_indices = env_ids * num_actors_per_env + 1
            ball_initial_states = all_root_states[ball_actor_indices].clone()
            ball_initial_states[:, 0] = self._initial_root_states[env_ids, 0] + 0.8
            ball_initial_states[:, 1] = self._initial_root_states[env_ids, 1] + 0.0
            ball_initial_states[:, 2] = 0.11
            ball_initial_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)  # Unit quaternion
            ball_initial_states[:, 7:13] = 0  # Zero velocity
            all_root_states[ball_actor_indices] = ball_initial_states
            
            # Use all actor indices that need to be reset
            all_actor_indices = torch.cat([humanoid_actor_indices, ball_actor_indices]).to(dtype=torch.int32)
        else:
            # No ball, use environment ID directly as actor index
            all_actor_indices = env_ids.to(dtype=torch.int32)
            # Update _actor_root_state_tensor
            all_root_states = gymtorch.wrap_tensor(self._actor_root_state_tensor)
            all_root_states[all_actor_indices] = self._initial_root_states[env_ids]
        
        # Apply state updates
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._actor_root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_actor_indices), len(all_actor_indices))
        if self._has_ball_asset:
            humanoid_actor_indices_for_dof = humanoid_actor_indices.to(dtype=torch.int32)
        else:
            humanoid_actor_indices_for_dof = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(humanoid_actor_indices_for_dof), len(humanoid_actor_indices_for_dof))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf

        # Log ball related metrics to wandb
        if self._has_ball_asset:
            self._log_ball_metrics()

        # print(self._target_states, self.debug_viz, self.viewer)
        # exit(1)
        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self, mode="rgb_array"):
        if mode == "rgb_array" and self.virtual_screen_capture:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            for i in range(self.num_envs):
                char_pos = self._root_states[i, 0:3].cpu().numpy()
                cam_pos = gymapi.Vec3(char_pos[0], char_pos[1] - 3.0, 1.0)
                cam_target = gymapi.Vec3(char_pos[0], char_pos[1], 1.0)
                self.gym.set_camera_location(self.cam_handles[i], self.envs[i], cam_pos, cam_target)

            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)

            self.gym.render_all_camera_sensors(self.sim)

            self.gym.start_access_image_tensors(self.sim)
            # Get camera image for the first environment
            cam_img_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[0], self.cam_handles[0], gymapi.IMAGE_COLOR
            )
            
            # Convert to torch then numpy (RGBA format)
            torch_cam_tensor = gymtorch.wrap_tensor(cam_img_tensor)
            img_raw = torch_cam_tensor.cpu().numpy()
            self.gym.end_access_image_tensors(self.sim)

            # Visualize the target pose
            # target world position (bottom and top of the line)
            def world_to_pixel(pos_world, view, proj, width, height):
                """
                pos_world: (3,)
                return: (u, v) pixel coordinate
                """
                p = np.ones(4)
                p[:3] = pos_world
                clip = (p @ view) @ proj
                # print(f"p:{p}, view: {view}, proj: {proj}")
                # print(f"p @ view: {p @ view}, clip: {clip}")
                # print("clip:", clip)
                if clip[3] <= 0:
                    return None  # behind camera
                ndc = clip[:3] / clip[3]   # [-1, 1]
                u = int((ndc[0] * 0.5 + 0.5) * width)
                v = int((1.0 - (ndc[1] * 0.5 + 0.5)) * height)
                return u, v
            H, W, _ = img_raw.shape
            view = self.gym.get_camera_view_matrix(
                self.sim, self.envs[0], self.cam_handles[0]
            )
            proj = self.gym.get_camera_proj_matrix(
                self.sim, self.envs[0], self.cam_handles[0]
            )
            # print(self._target_states[0])
            tx, ty, tz = self._target_states[0, :3].cpu().numpy()
            p_bot = world_to_pixel(
                np.array([tx, ty, 0.0]), view, proj, W, H
            )
            p_top = world_to_pixel(
                np.array([tx, ty, 2.0]), view, proj, W, H
            )
            # print(H, W, p_bot, p_top)
            # print(view)
            # print(proj)
            if p_bot is not None and p_top is not None:
                cv2.line(
                    img_raw,
                    p_bot,
                    p_top,
                    (255, 0, 0),   # 红色（RGB）
                    2
                )

            # Return RGB (remove Alpha channel)
            return img_raw[:, :, :3]

        # Use default rendering if Viewer is present
        if self.viewer:
            if self.camera_follow:
                self._update_camera()
            return super().render(mode)
        
        return None


    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_ball_body_ids_tensor(self, env_ptr, actor_handle):
        """Build ball body ID tensor"""
        body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, "ball")
        assert(body_id != -1), "Ball body 'ball' not found"
        body_ids = to_torch([body_id], device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
                              self._cam_prev_char_pos[1] - 3.0, 
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        # Visualize Target
        if self._target_states is not None:            
            for i in range(self.num_envs):
                target_pos = self._target_states[i, :3].cpu().numpy()
                p1 = gymapi.Vec3(target_pos[0], target_pos[1], 0.0)
                p2 = gymapi.Vec3(target_pos[0], target_pos[1], 2.0)
                color = gymapi.Vec3(1.0, 0.0, 0.0)
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p1, p2], [color])
                
        return

    def _init_headless_cameras(self):
        """Initialize camera sensors for headless mode"""
        if self.graphics_device_id == -1:
            print("Warning: graphics_device_id is -1, camera sensors cannot be created!")
            exit(1)
        self.cam_handles = []
        camera_props = gymapi.CameraProperties()
        camera_props.width = 640
        camera_props.height = 480
        camera_props.enable_tensors = True # Must be True to support fast reading

        for i in range(self.num_envs):
            cam_handle = self.gym.create_camera_sensor(self.envs[i], camera_props)
            self.cam_handles.append(cam_handle)
            
            char_pos = self._root_states[i, 0:3].cpu().numpy()
            cam_pos = gymapi.Vec3(char_pos[0], char_pos[1] - 3.0, 1.0)
            cam_target = gymapi.Vec3(char_pos[0], char_pos[1], 1.0)
            self.gym.set_camera_location(cam_handle, self.envs[i], cam_pos, cam_target)

    def _update_headless_camera(self):
        # Add safety check
        if not hasattr(self, '_headless_camera_handle') or self._headless_camera_handle == -1:
            return

        env_ptr = self.envs[0]
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()

        # Calculate camera position (keep fixed offset)
        cam_pos = gymapi.Vec3(char_root_pos[0], char_root_pos[1] - 3.0, 1.0)
        cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        
        # Use set_camera_location which is more robust than set_camera_transform in simple follow scenarios
        self.gym.set_camera_location(self._headless_camera_handle, env_ptr, cam_pos, cam_target)

    def _log_ball_metrics(self):
        """Calculate and log ball-related monitoring metrics to extras for wandb display"""
        if not hasattr(self, 'extras'):
            self.extras = {}
        
        # Initialize episode_cumulative dictionary (if not exists)
        if 'episode_cumulative' not in self.extras:
            self.extras['episode_cumulative'] = {}
        
        # Calculate ball relative position and velocity
        ball_pos = self._ball_root_states[:, 0:3]
        ball_vel = self._ball_root_states[:, 7:10]
        root_pos = self._root_states[:, 0:3]
        root_rot = self._root_states[:, 3:7]
        root_vel = self._root_states[:, 7:10]
        
        # Calculate ball relative position (in local coordinate system)
        heading_rot = calc_heading_quat_inv(root_rot)
        ball_rel_pos = ball_pos - root_pos
        local_ball_rel_pos = my_quat_rotate(heading_rot, ball_rel_pos)
        
        # 1. Ball distance (relative to robot root)
        ball_distance = torch.norm(local_ball_rel_pos, dim=1)
        
        # 2. Velocity consistency (difference between ball velocity and robot velocity)
        local_root_vel = my_quat_rotate(heading_rot, root_vel)
        local_ball_vel = my_quat_rotate(heading_rot, ball_vel)
        vel_diff = local_ball_vel - local_root_vel
        velocity_consistency = torch.norm(vel_diff, dim=1)
        
        # 3. Contact frequency (whether feet contact the ball)
        contact_frequency = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        if hasattr(self, '_contact_forces'):
            right_foot_id = self._key_body_ids[2]  # right_foot
            left_foot_id = self._key_body_ids[3]   # left_foot
            foot_contact_forces = self._contact_forces[:, [right_foot_id, left_foot_id], :]
            contact_threshold = 10.0  # 10N
            foot_contact_magnitude = torch.norm(foot_contact_forces, dim=2)  # [num_envs, 2]
            has_foot_contact = torch.any(foot_contact_magnitude > contact_threshold, dim=1)
            contact_frequency = has_foot_contact.float()
        
        # Add metrics to episode_cumulative (these values will be automatically logged to wandb at episode end)
        self.extras['episode_cumulative']['ball_distance'] = ball_distance
        self.extras['episode_cumulative']['velocity_consistency'] = velocity_consistency
        self.extras['episode_cumulative']['contact_frequency'] = contact_frequency
        
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    #dof_obs_size = 64
    #dof_offsets = [0, 3, 6, 9, 12, 13, 16, 19, 20, 23, 24, 27, 30, 31, 34]
    dof_obs_size = 52
    dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs, ball_root_states, target_states):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):  # false
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)  # quat -> 6d

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos)

    # Base observations: z height: 1, root rot (6d): 6, local root lin vel: 3, local root ang vel: 3
    # dof pos: 52, dof vel: 28, end effector pose: 12
    base_obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    
    # Add ball info
    # Check if ball_root_states is zero tensor (meaning no ball)
    ball_pos = ball_root_states[:, 0:3]
    ball_rot = ball_root_states[:, 3:7]
    ball_vel = ball_root_states[:, 7:10]
    ball_ang_vel = ball_root_states[:, 10:13]
    
    # Calculate ball relative position to humanoid root (in local coordinate system)
    ball_rel_pos = ball_pos - root_pos
    local_ball_rel_pos = my_quat_rotate(heading_rot, ball_rel_pos)
    
    # Ball linear velocity (in local coordinate system)
    local_ball_vel = my_quat_rotate(heading_rot, ball_vel)

    # Ball angular velocity (in local coordinate system)
    local_ball_ang_vel = my_quat_rotate(heading_rot, ball_ang_vel)
    local_ball_rot = quat_mul(heading_rot, ball_rot)
    target_rel_pos = target_states[:, 0:3] - root_pos
    local_target_rel_pos = my_quat_rotate(heading_rot, target_rel_pos)
    
    # Concatenate extended ball info: 
    # pos(3) + orientation(4) + lin_vel(3) + ang_vel(3) = 13
    ball_extended_obs = torch.cat((local_ball_rel_pos, local_ball_rot, local_ball_vel, local_ball_ang_vel), dim=-1)
    
    # Add target info: rel_pos(3)
    target_obs = local_target_rel_pos

    obs = torch.cat((base_obs, ball_extended_obs, target_obs), dim=-1)
    
    return obs



@torch.jit.script
def compute_AMP_ball_reward(tgt_vel, root_pos, root_vel, ball_pos, ball_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    tgt_pos = torch.tensor([10.0, 10.0], device=tgt_vel.device)
    tgt_ball_dir = tgt_pos - ball_pos[..., 0:2]
    tgt_ball_dir = tgt_ball_dir / (torch.norm(tgt_ball_dir, dim=-1, keepdim=True) + 1e-6)

    tgt_vel_mag = torch.norm(tgt_vel)
    ball_rel_pos = ball_pos - root_pos
    ball_rel_dir = ball_rel_pos / (torch.norm(ball_rel_pos, dim=1, keepdim=True) + 1e-6)
    rwd_cv = torch.exp(-1.5 * (torch.clamp(tgt_vel_mag - torch.sum(root_vel * ball_rel_dir, dim=1), min=0.0)**2))
    rwd_cp = torch.exp(-0.5 * (torch.norm(ball_rel_pos, dim=1) ** 2))
    rwd_bv = torch.exp(-1.0 * ((tgt_vel_mag - torch.sum(ball_vel[..., 0:2] * tgt_ball_dir, dim=1))**2))
    rwd_bp = torch.exp(-0.5 * (torch.norm(ball_pos[..., 0:2] - tgt_pos, dim=1) ** 2))

    reward = 0.0 * rwd_cv + 0.4 * rwd_cp + 0.2 * rwd_bv + 0.4 * rwd_bp
    return reward

@torch.jit.script
def compute_AMP_ball_reward_target(target_speed, root_pos, root_vel, ball_pos, ball_vel, target_pos):
    # type: (float, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    ball_to_target = target_pos - ball_pos
    dist_ball_target = torch.norm(ball_to_target, dim=1, keepdim=True)
    d_star = ball_to_target / (dist_ball_target + 1e-6)
    
    # Calculate direction from char to ball (d_ball_t)
    char_to_ball = ball_pos - root_pos
    dist_char_ball = torch.norm(char_to_ball, dim=1, keepdim=True)
    d_ball = char_to_ball / (dist_char_ball + 1e-6)
    
    v_star = target_speed # Target speed, usually 1 m/s
    
    v_com_projected = torch.sum(root_vel * d_ball, dim=1, keepdim=True)
    rwd_cv = torch.exp(-1.5 * (torch.clamp(v_star - v_com_projected, min=0.0)**2))

    rwd_cp = torch.exp(-0.5 * (dist_char_ball**2))
    
    v_ball_projected = torch.sum(ball_vel * d_star, dim=1, keepdim=True)
    # rwd_bv = torch.exp(-1.0 * (torch.clamp(v_star - v_ball_projected, min=0.0)**2))
    rwd_bv = torch.exp(-1.0 * ((v_star - v_ball_projected)**2))
    
    dist_char_target = torch.norm(target_pos - root_pos, dim=1, keepdim=True)
    rwd_bp = torch.exp(-0.5 * (dist_char_target**2))
    
    reward = 0.0 * rwd_cv.squeeze() + 0.2 * rwd_cp.squeeze() + 0.3 * rwd_bv.squeeze() + 0.5 * rwd_bp.squeeze()
    
    return reward

@torch.jit.script
def compute_humanoid_reward(obs_buf, task_speed, task_speed_mul, root_states, ball_root_states, foot_contact_forces, target_states, enable_ball_reward, enable_target):
    # type: (Tensor, float, float, Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor

    root_local_x_speed = obs_buf[:, 7]
    speed_rwd = torch.exp(-(root_local_x_speed - task_speed)**2.0) * task_speed_mul  # Default speed reward
    reward = speed_rwd

    if enable_ball_reward:
        if enable_target:
            return compute_AMP_ball_reward_target(
                target_speed=task_speed,
                root_pos=root_states[:, 0:3],
                root_vel=root_states[:, 7:10],
                ball_pos=ball_root_states[:, 0:3],
                ball_vel=ball_root_states[:, 7:10],
                target_pos=target_states
            )
        else:
            num_envs = root_states.shape[0]
            tgt_vel = torch.zeros((num_envs, 3), device=root_states.device)
            tgt_vel[:, 0] = task_speed
            return compute_AMP_ball_reward(
                tgt_vel=tgt_vel,
                root_pos=root_states[:, 0:3],
                root_vel=root_states[:, 7:10],
                ball_pos=ball_root_states[:, 0:3],
                ball_vel=ball_root_states[:, 7:10]
            )
    
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_height,
                           ball_root_states, target_states):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float, Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    # Check if ball reached target
    if ball_root_states is not None and target_states is not None:
        ball_pos = ball_root_states[:, 0:3]
        target_pos = target_states[:, 0:3]
        
        dist = torch.norm(ball_pos - target_pos, dim=1)
        success = dist < 0.5
        
        # Reset if success
        reset = torch.where(success, torch.ones_like(reset), reset)

    return reset, terminated