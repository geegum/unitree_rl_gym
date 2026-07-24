import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import (
    get_axis_params,
    quat_rotate_inverse,
    to_torch,
)

from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.utils.isaacgym_utils import (
    get_euler_xyz as get_euler_xyz_in_tensor,
)


class G1BalanceV0Robot(G1Robot):
    """G1 controller with 12 policy actions and 29 torque outputs."""

    LEG_DOF_NAMES = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    )

    def _get_dof_indices(self, names):
        missing = [name for name in names if name not in self.dof_names]

        if missing:
            raise RuntimeError(f"Missing DOF names: {missing}")

        return torch.tensor(
            [self.dof_names.index(name) for name in names],
            dtype=torch.long,
            device=self.device,
        )

    def _init_buffers(self):
        """Create 12-action buffers and 29-DOF control buffers."""

        actor_root_state = self.gym.acquire_actor_root_state_tensor(
            self.sim
        )
        dof_state_tensor = self.gym.acquire_dof_state_tensor(
            self.sim
        )
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(
            self.sim
        )

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.dof_pos = self.dof_state.view(
            self.num_envs, self.num_dof, 2
        )[..., 0]

        self.dof_vel = self.dof_state.view(
            self.num_envs, self.num_dof, 2
        )[..., 1]

        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]

        self.contact_forces = gymtorch.wrap_tensor(
            net_contact_forces
        ).view(self.num_envs, -1, 3)

        if self.num_dof != 29:
            raise RuntimeError(
                f"Expected 29 DOFs, but loaded {self.num_dof}"
            )

        if self.num_actions != 12:
            raise RuntimeError(
                f"Expected 12 actions, but configured "
                f"{self.num_actions}"
            )

        self.leg_dof_indices = self._get_dof_indices(
            self.LEG_DOF_NAMES
        )

        all_indices = torch.arange(
            self.num_dof,
            dtype=torch.long,
            device=self.device,
        )

        fixed_mask = torch.ones(
            self.num_dof,
            dtype=torch.bool,
            device=self.device,
        )

        fixed_mask[self.leg_dof_indices] = False
        self.fixed_dof_indices = all_indices[fixed_mask]

        if self.fixed_dof_indices.numel() != 17:
            raise RuntimeError(
                "Expected 17 fixed upper-body DOFs, got "
                f"{self.fixed_dof_indices.numel()}"
            )

        self.common_step_counter = 0
        self.extras = {}

        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx),
            device=self.device,
        ).repeat((self.num_envs, 1))

        self.forward_vec = to_torch(
            [1.0, 0.0, 0.0],
            device=self.device,
        ).repeat((self.num_envs, 1))

        # Isaac Gym receives 29 torques.
        self.torques = torch.zeros(
            self.num_envs,
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # PD gains are defined for all 29 joints.
        self.p_gains = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.d_gains = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # PPO still produces only 12 actions.
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.last_root_vel = torch.zeros_like(
            self.root_states[:, 7:13]
        )

        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
            ],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat,
            self.root_states[:, 7:10],
        )

        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat,
            self.root_states[:, 10:13],
        )

        self.projected_gravity = quat_rotate_inverse(
            self.base_quat,
            self.gravity_vec,
        )

        self.default_dof_pos = torch.zeros(
            self.num_dof,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        configured_angles = (
            self.cfg.init_state.default_joint_angles
        )

        for i, name in enumerate(self.dof_names):
            if name not in configured_angles:
                raise KeyError(
                    f"Default angle missing for joint: {name}"
                )

            self.default_dof_pos[i] = configured_angles[name]

            gain_found = False

            for gain_name, stiffness in (
                self.cfg.control.stiffness.items()
            ):
                if gain_name in name:
                    if gain_name not in self.cfg.control.damping:
                        raise KeyError(
                            f"Damping missing for: {gain_name}"
                        )

                    self.p_gains[i] = stiffness
                    self.d_gains[i] = (
                        self.cfg.control.damping[gain_name]
                    )

                    gain_found = True
                    break

            if not gain_found:
                raise KeyError(
                    f"PD gain missing for joint: {name}"
                )

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.joint_pos_targets = self.default_dof_pos.repeat(
            self.num_envs, 1
        )

        self.noise_scale_vec = self._get_noise_scale_vec(
            self.cfg
        )

        # G1 foot rigid-body state buffers.
        self._init_foot()

        print("\n[G1BalanceV0] Control mapping initialized")
        print(f"  Policy actions: {self.num_actions}")
        print(f"  Simulated DOFs: {self.num_dof}")

        print(
            "  Leg indices:",
            self.leg_dof_indices.detach().cpu().tolist(),
        )

        print(
            "  Fixed indices:",
            self.fixed_dof_indices.detach().cpu().tolist(),
        )

        print(f"  Torque buffer: {tuple(self.torques.shape)}")
        print(f"  Action buffer: {tuple(self.actions.shape)}")

    def _compute_torques(self, actions):
        """Expand 12 leg actions into 29 joint PD torques."""

        if actions.ndim != 2 or actions.shape[1] != 12:
            raise RuntimeError(
                "Expected actions with shape [num_envs, 12], "
                f"got {tuple(actions.shape)}"
            )

        if self.cfg.control.control_type != "P":
            raise RuntimeError(
                "G1BalanceV0 supports only P control"
            )

        actions_scaled = (
            actions * self.cfg.control.action_scale
        )

        # Reset all 29 targets to the configured home posture.
        self.joint_pos_targets[:] = self.default_dof_pos

        # Add RL actions only to the twelve leg joints.
        self.joint_pos_targets[
            :, self.leg_dof_indices
        ] += actions_scaled

        torques = (
            self.p_gains
            * (self.joint_pos_targets - self.dof_pos)
            - self.d_gains * self.dof_vel
        )

        return torch.clip(
            torques,
            -self.torque_limits,
            self.torque_limits,
        )

    def _get_noise_scale_vec(self, cfg):
        """Noise layout for the 45-dimensional observation."""

        noise_vec = torch.zeros_like(self.obs_buf[0])

        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level

        # Base angular velocity
        noise_vec[0:3] = (
            noise_scales.ang_vel
            * noise_level
            * self.obs_scales.ang_vel
        )

        # Projected gravity
        noise_vec[3:6] = (
            noise_scales.gravity * noise_level
        )

        # Commands: no noise
        noise_vec[6:9] = 0.0

        # Leg positions
        noise_vec[9:21] = (
            noise_scales.dof_pos
            * noise_level
            * self.obs_scales.dof_pos
        )

        # Leg velocities
        noise_vec[21:33] = (
            noise_scales.dof_vel
            * noise_level
            * self.obs_scales.dof_vel
        )

        # Previous actions: no noise
        noise_vec[33:45] = 0.0

        return noise_vec

    def compute_observations(self):
        """Policy sees only base state, command and leg states."""

        leg_pos = self.dof_pos[:, self.leg_dof_indices]
        leg_vel = self.dof_vel[:, self.leg_dof_indices]

        default_leg_pos = self.default_dof_pos[
            :, self.leg_dof_indices
        ]

        policy_observation = torch.cat(
            (
                self.base_ang_vel
                * self.obs_scales.ang_vel,

                self.projected_gravity,

                self.commands[:, :3]
                * self.commands_scale,

                (leg_pos - default_leg_pos)
                * self.obs_scales.dof_pos,

                leg_vel
                * self.obs_scales.dof_vel,

                self.actions,
            ),
            dim=-1,
        )

        if policy_observation.shape[1] != 45:
            raise RuntimeError(
                "Policy observation must have 45 elements, "
                f"got {policy_observation.shape[1]}"
            )

        self.obs_buf = policy_observation

        self.privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel
                * self.obs_scales.lin_vel,

                policy_observation,
            ),
            dim=-1,
        )

        if self.privileged_obs_buf.shape[1] != 48:
            raise RuntimeError(
                "Privileged observation must have 48 elements, "
                f"got {self.privileged_obs_buf.shape[1]}"
            )

        if self.add_noise:
            self.obs_buf += (
                2.0 * torch.rand_like(self.obs_buf) - 1.0
            ) * self.noise_scale_vec

    def _reward_upper_body_pose(self):
        """Keep waist and both arms near their home posture."""

        position_error = (
            self.dof_pos[:, self.fixed_dof_indices]
            - self.default_dof_pos[
                :, self.fixed_dof_indices
            ]
        )

        return torch.sum(
            torch.square(position_error),
            dim=1,
        )
