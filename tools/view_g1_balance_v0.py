import isaacgym
from isaacgym import gymtorch

import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


TASK_NAME = "g1_balance_v0"


def set_exact_home_pose(env, env_ids):
    """Reset selected environments to the exact configured home pose."""

    if env_ids.numel() == 0:
        return

    ids_long = env_ids.to(device=env.device, dtype=torch.long)
    ids_int32 = ids_long.to(dtype=torch.int32)

    # Exact configured joint pose, zero joint velocity.
    env.dof_pos[ids_long] = env.default_dof_pos
    env.dof_vel[ids_long] = 0.0

    # Exact base pose at each environment origin.
    env.root_states[ids_long] = env.base_init_state
    env.root_states[ids_long, :3] += env.env_origins[ids_long]
    env.root_states[ids_long, 7:13] = 0.0

    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(ids_int32),
        len(ids_int32),
    )

    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(ids_int32),
        len(ids_int32),
    )


def main():
    args = get_args()

    # This script is always a one-robot visual inspection.
    args.task = TASK_NAME
    args.num_envs = 1
    args.headless = False

    env_cfg, _ = task_registry.get_cfgs(name=TASK_NAME)

    env_cfg.env.num_envs = 1
    env_cfg.env.test = True

    # Deterministic visual inspection.
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False

    # Camera for one robot at the origin.
    env_cfg.viewer.pos = [3.0, -3.0, 1.8]
    env_cfg.viewer.lookat = [0.0, 0.0, 0.8]

    env, _ = task_registry.make_env(
        name=TASK_NAME,
        args=args,
        env_cfg=env_cfg,
    )

    env.reset()

    all_env_ids = torch.arange(
        env.num_envs,
        device=env.device,
        dtype=torch.long,
    )
    set_exact_home_pose(env, all_env_ids)

    # Policy actions remain exactly zero.
    actions = torch.zeros(
        env.num_envs,
        env.num_actions,
        device=env.device,
        dtype=torch.float,
    )

    print()
    print("[G1BalanceV0 Viewer]")
    print("  Policy learning: disabled")
    print("  Leg actions: all zero")
    print("  Applied torque dimensions:", env.num_dof)
    print("  Close window or press Esc to exit")
    print()

    step_count = 0
    reset_count = 0

    try:
        with torch.inference_mode():
            while True:
                _, _, rewards, dones, _ = env.step(actions)

                step_count += 1

                reset_ids = torch.nonzero(
                    dones,
                    as_tuple=False,
                ).flatten()

                if reset_ids.numel() > 0:
                    reset_count += reset_ids.numel()
                    set_exact_home_pose(env, reset_ids)

                    print(
                        f"[reset] count={reset_count}, "
                        f"step={step_count}"
                    )

                if step_count % 250 == 0:
                    base_z = env.root_states[0, 2].item()
                    roll = env.rpy[0, 0].item()
                    pitch = env.rpy[0, 1].item()
                    yaw = env.rpy[0, 2].item()

                    print(
                        f"[state] step={step_count}, "
                        f"base_z={base_z:.3f}, "
                        f"rpy=({roll:.3f}, "
                        f"{pitch:.3f}, {yaw:.3f}), "
                        f"reward={rewards[0].item():.4f}"
                    )

    except (KeyboardInterrupt, SystemExit):
        print("\nViewer stopped.")

    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)

        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
