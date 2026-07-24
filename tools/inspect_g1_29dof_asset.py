from pathlib import Path

from isaacgym import gymapi


EXPECTED_DOF_COUNT = 29
EXPECTED_RIGHT_ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    asset_root = repo_root / "resources" / "robots" / "g1_description"
    asset_file = "g1_29dof_rev_1_0.urdf"

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 0.005
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True

    sim = gym.create_sim(
        0,
        -1,
        gymapi.SIM_PHYSX,
        sim_params,
    )

    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")

    options = gymapi.AssetOptions()
    options.fix_base_link = False
    options.collapse_fixed_joints = True
    options.replace_cylinder_with_capsule = False
    options.flip_visual_attachments = False
    options.disable_gravity = False

    asset = gym.load_asset(
        sim,
        str(asset_root),
        asset_file,
        options,
    )

    if asset is None:
        gym.destroy_sim(sim)
        raise RuntimeError("Failed to load G1 29-DOF asset")

    dof_names = list(gym.get_asset_dof_names(asset))
    body_names = list(gym.get_asset_rigid_body_names(asset))

    print("Asset root:", asset_root)
    print("Asset file:", asset_file)
    print("DOF count:", len(dof_names))
    print("Rigid body count:", len(body_names))

    print("\nDOF names:")
    for i, name in enumerate(dof_names):
        print(f"{i:2d}: {name}")

    print("\nRequired links:")

    end_effector_name = None
    for candidate in ("right_rubber_hand", "right_wrist_yaw_link"):
        if candidate in body_names:
            end_effector_name = candidate
            break

    print("Selected end-effector:", end_effector_name)

    assert len(dof_names) == EXPECTED_DOF_COUNT, (
        f"Expected {EXPECTED_DOF_COUNT} DOFs, "
        f"but Isaac Gym loaded {len(dof_names)}"
    )

    missing_arm_joints = [
        name for name in EXPECTED_RIGHT_ARM
        if name not in dof_names
    ]

    assert not missing_arm_joints, (
        f"Missing right-arm joints: {missing_arm_joints}"
    )

    assert end_effector_name is not None, (
        "Neither right_rubber_hand nor right_wrist_yaw_link was loaded"
    )

    print("\nIsaac Gym asset test: PASS")
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
