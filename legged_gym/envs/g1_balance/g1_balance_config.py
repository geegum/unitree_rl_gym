from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO


class G1BalanceV0Cfg(G1RoughCfg):
    """G1 standing balance with 12 leg actions and fixed upper-body PD."""

    class env(G1RoughCfg.env):
        num_envs = 4096

        # 3 base angular velocity
        # 3 projected gravity
        # 3 commands
        # 12 leg positions
        # 12 leg velocities
        # 12 previous actions
        num_observations = 45

        # Policy observation + 3 base linear velocities
        num_privileged_obs = 48

        # Policy controls only both legs.
        num_actions = 12

        episode_length_s = 20
        test = False

    class terrain(G1RoughCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False

    class commands(G1RoughCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 10.0

        class ranges(G1RoughCfg.commands.ranges):
            # Standing task: no walking command.
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class init_state(G1RoughCfg.init_state):
        pos = [0.0, 0.0, 0.8]

        default_joint_angles = {
            # Left leg
            "left_hip_pitch_joint": -0.10,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.30,
            "left_ankle_pitch_joint": -0.20,
            "left_ankle_roll_joint": 0.0,

            # Right leg
            "right_hip_pitch_joint": -0.10,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.30,
            "right_ankle_pitch_joint": -0.20,
            "right_ankle_roll_joint": 0.0,

            # Waist: fixed by PD
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,

            # Left arm: fixed by PD
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,

            # Right arm: fixed by PD
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

    class control(G1RoughCfg.control):
        control_type = "P"

        stiffness = {
            # Legs
            "hip_yaw": 100.0,
            "hip_roll": 100.0,
            "hip_pitch": 100.0,
            "knee": 150.0,
            "ankle_pitch": 40.0,
            "ankle_roll": 40.0,

            # Waist
            "waist_yaw": 300.0,
            "waist_roll": 300.0,
            "waist_pitch": 300.0,

            # Arms
            "shoulder": 80.0,
            "elbow": 80.0,
            "wrist": 40.0,
        }

        damping = {
            # Legs
            "hip_yaw": 2.0,
            "hip_roll": 2.0,
            "hip_pitch": 2.0,
            "knee": 4.0,
            "ankle_pitch": 2.0,
            "ankle_roll": 2.0,

            # Waist
            "waist_yaw": 3.0,
            "waist_roll": 3.0,
            "waist_pitch": 3.0,

            # Arms
            "shoulder": 3.0,
            "elbow": 3.0,
            "wrist": 1.5,
        }

        # Applied only to the 12 leg actions.
        action_scale = 0.25
        decimation = 4

    class asset(G1RoughCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            "g1_description/g1_29dof_rev_1_0.urdf"
        )

        name = "g1_balance_v0"
        foot_name = "ankle_roll"

        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]

        collapse_fixed_joints = True
        self_collisions = 0
        flip_visual_attachments = False

    class domain_rand(G1RoughCfg.domain_rand):
        # Disable randomization during the first controller validation.
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class rewards(G1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        only_positive_rewards = True

        class scales(G1RoughCfg.rewards.scales):
            termination = -2.0

            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5

            lin_vel_z = -2.0
            ang_vel_xy = -0.10
            orientation = -2.0
            base_height = -10.0

            torques = -1.0e-5
            dof_acc = -2.5e-7
            dof_vel = -1.0e-3
            action_rate = -0.01

            dof_pos_limits = -5.0
            collision = -1.0
            alive = 0.20

            stand_still = -0.10
            upper_body_pose = -0.50

            # Disable walking-cycle rewards.
            feet_air_time = 0.0
            feet_swing_height = 0.0
            contact = 0.0
            hip_pos = -0.20
            contact_no_vel = -0.20


class G1BalanceV0CfgPPO(G1RoughCfgPPO):
    class policy(G1RoughCfgPPO.policy):
        init_noise_std = 0.4
        actor_hidden_dims = [128, 128]
        critic_hidden_dims = [128, 128]

        rnn_type = "lstm"
        rnn_hidden_size = 128
        rnn_num_layers = 1

    class runner(G1RoughCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        experiment_name = "g1_balance_v0"
        run_name = ""
