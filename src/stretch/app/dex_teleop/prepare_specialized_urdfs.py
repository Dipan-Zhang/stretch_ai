# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import argparse
import pathlib
import pprint
from copy import deepcopy

import stretch.app.dex_teleop.dex_teleop_parameters as dt
import numpy as np
import stretch.motion.constants as constants
from urdf_parser_py import urdf as ud


def save_urdf_file(robot, file_name):
    urdf_string = robot.to_xml_string()
    print("Saving new URDF file to", file_name)
    fid = open(file_name, "w")
    fid.write(urdf_string)
    fid.close()
    print("Finished saving")


def parse_joint_limit(value):
    """Parse joint limit value from string. 'None' or empty string means use original limit."""
    if value is None or value == "" or value.lower() == "none":
        return None
    return float(value)


# Global variable defining all custom joint limit arguments
CUSTOM_JOINT_LIMIT_ARGS = [
    {
        "name": "--base-translation-lower",
        "dest": "base_translation_lower",
        "help": "Lower limit for joint_mobile_base_translation (meters).",
    },
    {
        "name": "--base-translation-upper",
        "dest": "base_translation_upper",
        "help": "Upper limit for joint_mobile_base_translation (meters).",
    },
    {
        "name": "--base-rotation-lower",
        "dest": "base_rotation_lower",
        "help": "Lower limit for joint_mobile_base_rotation (radians).",
    },
    {
        "name": "--base-rotation-upper",
        "dest": "base_rotation_upper",
        "help": "Upper limit for joint_mobile_base_rotation (radians).",
    },
    {
        "name": "--lift-lower",
        "dest": "lift_lower",
        "help": "Lower limit for joint_lift (meters).",
    },
    {
        "name": "--lift-upper",
        "dest": "lift_upper",
        "help": "Upper limit for joint_lift (meters).",
    },
    {
        "name": "--arm-l0-lower",
        "dest": "arm_l0_lower",
        "help": "Lower limit for joint_arm_l0 (meters).",
    },
    {
        "name": "--arm-l0-upper",
        "dest": "arm_l0_upper",
        "help": "Upper limit for joint_arm_l0 (meters).",
    },
    {
        "name": "--wrist-yaw-lower",
        "dest": "wrist_yaw_lower",
        "help": "Lower limit for joint_wrist_yaw (radians).",
    },
    {
        "name": "--wrist-yaw-upper",
        "dest": "wrist_yaw_upper",
        "help": "Upper limit for joint_wrist_yaw (radians).",
    },
    {
        "name": "--wrist-pitch-lower",
        "dest": "wrist_pitch_lower",
        "help": "Lower limit for joint_wrist_pitch (radians).",
    },
    {
        "name": "--wrist-pitch-upper",
        "dest": "wrist_pitch_upper",
        "help": "Upper limit for joint_wrist_pitch (radians).",
    },
    {
        "name": "--wrist-roll-lower",
        "dest": "wrist_roll_lower",
        "help": "Lower limit for joint_wrist_roll (radians).",
    },
    {
        "name": "--wrist-roll-upper",
        "dest": "wrist_roll_upper",
        "help": "Upper limit for joint_wrist_roll (radians).",
    },
]


def get_urdf_filename(use_on_robot, urdf_path):
    """Get URDF filename based on configuration."""
    if use_on_robot:
        import stretch_body.hello_utils as hu
        calibration_dir = pathlib.Path(hu.get_fleet_directory()) / "exported_urdf"
        urdf_path = calibration_dir / "stretch.urdf"
        return str(urdf_path.absolute())
    else:
        return urdf_path if urdf_path else constants.MANIP_STRETCH_URDF


def get_joint_limits(use_original_limits, wrist_pitch_lower_limit):
    """Get joint limits configuration based on flags."""
    if use_original_limits:
        # Beware of gimbal lock if joint_wrist_pitch is too close to -90 deg
        return {
            "joint_mobile_base_translation": (None, None),
            "joint_mobile_base_rotation": (None, None),
            "joint_lift": (None, None),
            "joint_arm_l0": (None, None),
            "joint_wrist_yaw": (None, None),
            "joint_wrist_pitch": (wrist_pitch_lower_limit, None),
            "joint_wrist_roll": (None, None),
        }
    else:
        return {
            "joint_mobile_base_translation": (-0.25, 0.25),
            "joint_mobile_base_rotation": (-(np.pi / 2.0), np.pi / 2.0),
            "joint_lift": (0.01, 1.09),
            "joint_arm_l0": (0.01, 0.48),
            "joint_wrist_yaw": (-(np.pi / 4.0), np.pi),
            "joint_wrist_pitch": (-0.9 * (np.pi / 2.0), np.pi / 20.0),
            "joint_wrist_roll": (-(np.pi / 2.0), np.pi / 2.0),
        }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare specialized URDF files for Stretch robot IK solvers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # URDF input/output configuration
    parser.add_argument(
        "--use-on-robot",
        default=True,
        type=bool,
        help="Use robot-specific URDF path from fleet directory. If not set, use --urdf-path.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="",
        help="Path to input URDF file (used when --use-on-robot is False).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output URDF files.",
    )
    
    # Joint limits configuration
    parser.add_argument(
        "--use-original-limits",
        action="store_true",
        default=True,
        help="Use original URDF joint limits (with optional wrist_pitch override). "
             "If False, use conservative custom limits.",
    )
    parser.add_argument(
        "--wrist-pitch-lower-limit",
        type=float,
        default=None,
        help="Override lower limit for joint_wrist_pitch (in radians). "
             "If None and --use-original-limits, uses dt.wrist_pitch_lower_limit.",
    )
    
    # Custom joint limits (for when --use-original-limits is False)
    joint_limit_group = parser.add_argument_group(
        "Custom Joint Limits",
        "Override individual joint limits when --use-original-limits is False. "
        "Use 'None' or leave empty to use default conservative limits."
    )
    
    # Add all custom joint limit arguments from global variable
    for arg_def in CUSTOM_JOINT_LIMIT_ARGS:
        joint_limit_group.add_argument(
            arg_def["name"],
            dest=arg_def.get("dest", arg_def["name"].lstrip("--").replace("-", "_")),
            type=parse_joint_limit,
            default=None,
            help=arg_def["help"],
        )
    
    return parser.parse_args()


def main():
    """Main function to prepare specialized URDF files."""
    args = parse_args()
    
    # Get URDF filename
    urdf_filename = get_urdf_filename(args.use_on_robot, args.urdf_path)
    
    # Get joint limits configuration
    wrist_pitch_lower = args.wrist_pitch_lower_limit
    if wrist_pitch_lower is None and args.use_original_limits:
        wrist_pitch_lower = dt.wrist_pitch_lower_limit
    
    ik_joint_limits = get_joint_limits(args.use_original_limits, wrist_pitch_lower)
    
    # Override with custom limits if provided and not using original limits
    if not args.use_original_limits:
        custom_limits = {
            "joint_mobile_base_translation": (
                args.base_translation_lower if args.base_translation_lower is not None else ik_joint_limits["joint_mobile_base_translation"][0],
                args.base_translation_upper if args.base_translation_upper is not None else ik_joint_limits["joint_mobile_base_translation"][1],
            ),
            "joint_mobile_base_rotation": (
                args.base_rotation_lower if args.base_rotation_lower is not None else ik_joint_limits["joint_mobile_base_rotation"][0],
                args.base_rotation_upper if args.base_rotation_upper is not None else ik_joint_limits["joint_mobile_base_rotation"][1],
            ),
            "joint_lift": (
                args.lift_lower if args.lift_lower is not None else ik_joint_limits["joint_lift"][0],
                args.lift_upper if args.lift_upper is not None else ik_joint_limits["joint_lift"][1],
            ),
            "joint_arm_l0": (
                args.arm_l0_lower if args.arm_l0_lower is not None else ik_joint_limits["joint_arm_l0"][0],
                args.arm_l0_upper if args.arm_l0_upper is not None else ik_joint_limits["joint_arm_l0"][1],
            ),
            "joint_wrist_yaw": (
                args.wrist_yaw_lower if args.wrist_yaw_lower is not None else ik_joint_limits["joint_wrist_yaw"][0],
                args.wrist_yaw_upper if args.wrist_yaw_upper is not None else ik_joint_limits["joint_wrist_yaw"][1],
            ),
            "joint_wrist_pitch": (
                args.wrist_pitch_lower if args.wrist_pitch_lower is not None else ik_joint_limits["joint_wrist_pitch"][0],
                args.wrist_pitch_upper if args.wrist_pitch_upper is not None else ik_joint_limits["joint_wrist_pitch"][1],
            ),
            "joint_wrist_roll": (
                args.wrist_roll_lower if args.wrist_roll_lower is not None else ik_joint_limits["joint_wrist_roll"][0],
                args.wrist_roll_upper if args.wrist_roll_upper is not None else ik_joint_limits["joint_wrist_roll"][1],
            ),
        }
        ik_joint_limits = custom_limits
        print("Setting custom joint limits with ik_joint_limits =")
        pprint.pprint(ik_joint_limits)
    
    # Define non-fixed joints
    non_fixed_joints = [
        "joint_lift",
        "joint_arm_l0",
        "joint_arm_l1",
        "joint_arm_l2",
        "joint_arm_l3",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
    ]
    
    # Load and store the original uncalibrated URDF.
    print()
    print("Loading URDF from:")
    print(urdf_filename)
    print("The specialized URDFs will be derived from this URDF.")
    robot = ud.Robot.from_xml_file(urdf_filename)

    # Change any joint that should be immobile for end effector IK into a fixed joint
    for j in robot.joint_map.keys():
        if j not in non_fixed_joints:
            joint = robot.joint_map[j]
            # print('(joint name, joint type) =', (joint.name, joint.type))
            joint.type = "fixed"

    # Replace telescoping arm with a single prismatic joint

    # arm joints from proximal to distal
    all_arm_joints = ["joint_arm_l4", "joint_arm_l3", "joint_arm_l2", "joint_arm_l1", "joint_arm_l0"]

    prismatic_arm_joints = all_arm_joints[1:]

    removed_arm_joints = all_arm_joints[1:-1]

    xyz_total = np.array([0.0, 0.0, 0.0])
    limit_upper_total = 0.0

    for j in prismatic_arm_joints:
        joint = robot.joint_map[j]
        # print(j + ' =', joint)
        xyz = joint.origin.xyz
        # print('xyz =', xyz)
        xyz_total = xyz_total + xyz
        limit_upper = joint.limit.upper
        # print('limit_upper =', limit_upper)
        limit_upper_total = limit_upper_total + limit_upper

    # print('xyz_total =', xyz_total)
    # print('limit_upper_total =', limit_upper_total)

    proximal_arm_joint = robot.joint_map[all_arm_joints[0]]
    near_proximal_arm_joint = robot.joint_map[all_arm_joints[1]]
    near_distal_arm_joint = robot.joint_map[all_arm_joints[-2]]
    distal_arm_joint = robot.joint_map[all_arm_joints[-1]]

    # Directly connect the proximal and distal parts of the arm
    distal_arm_joint.parent = near_proximal_arm_joint.parent

    # Make the distal prismatic joint act like the full arm
    distal_arm_joint.origin.xyz = xyz_total
    distal_arm_joint.limit.upper = limit_upper_total

    # Make the telescoping joints in between immobile
    for j in removed_arm_joints:
        joint = robot.joint_map[j]
        joint.type = "fixed"

    robot_rotary = robot
    robot_prismatic = deepcopy(robot)

    ###############################################
    # ADD VIRTUAL ROTARY JOINT FOR MOBILE BASE

    # Add a virtual base link
    link_virtual_base_rotary = ud.Link(
        name="virtual_base", visual=None, inertial=None, collision=None, origin=None
    )

    # Add rotary joint for the mobile base
    origin_rotary = ud.Pose(xyz=[0, 0, 0], rpy=[0, 0, 0])

    limit_rotary = ud.JointLimit(effort=10, velocity=1, lower=-np.pi, upper=np.pi)

    joint_mobile_base_rotation = ud.Joint(
        name="joint_mobile_base_rotation",
        parent="virtual_base",
        child="base_link",
        joint_type="revolute",
        axis=[0, 0, 1],
        origin=origin_rotary,
        limit=limit_rotary,
        dynamics=None,
        safety_controller=None,
        calibration=None,
        mimic=None,
    )

    robot_rotary.add_link(link_virtual_base_rotary)
    robot_rotary.add_joint(joint_mobile_base_rotation)
    ###############################################

    ###############################################
    # ADD VIRTUAL PRISMATIC JOINT FOR MOBILE BASE

    # Add a virtual base link
    link_virtual_base_prismatic = ud.Link(
        name="virtual_base", visual=None, inertial=None, collision=None, origin=None
    )

    # Add rotary joint for the mobile base
    origin_prismatic = ud.Pose(xyz=[0, 0, 0], rpy=[0, 0, 0])

    limit_prismatic = ud.JointLimit(effort=10, velocity=1, lower=-1.0, upper=1.0)

    joint_mobile_base_translation = ud.Joint(
        name="joint_mobile_base_translation",
        parent="virtual_base",
        child="base_link",
        joint_type="prismatic",
        axis=[1, 0, 0],
        origin=origin_prismatic,
        limit=limit_prismatic,
        dynamics=None,
        safety_controller=None,
        calibration=None,
        mimic=None,
    )

    robot_prismatic.add_link(link_virtual_base_prismatic)
    robot_prismatic.add_joint(joint_mobile_base_translation)

    ###############################################

    # When specified, this sets more conservative joint limits than the
    # original URDF. Joint limits that are outside the originally
    # permitted range are clipped to the original range. Joint limits
    # with a value of None are set to the original limit.
    for robot in [robot_rotary, robot_prismatic]:
        for j in ik_joint_limits:
            joint = robot.joint_map.get(j, None)
            if joint is not None:

                original_upper = joint.limit.upper
                requested_upper = ik_joint_limits[j][1]
                # print()
                # print('joint =', j)
                # print('original_upper =', original_upper)
                # print('requested_upper =', requested_upper)
                if requested_upper is not None:
                    new_upper = min(requested_upper, original_upper)
                    # print('new_upper =', new_upper)
                    robot.joint_map[j].limit.upper = new_upper
                    # print()

                original_lower = joint.limit.lower
                requested_lower = ik_joint_limits[j][0]
                if requested_lower is not None:
                    new_lower = max(requested_lower, original_lower)
                    robot.joint_map[j].limit.lower = new_lower

    # print('************************************************')
    # print('after adding link and joint: robot =', robot)
    # print('************************************************')
    
    # Prepare output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print()
    save_urdf_file(robot_rotary, str(output_dir / "stretch_base_rotation_ik.urdf"))
    save_urdf_file(robot_prismatic, str(output_dir / "stretch_base_translation_ik.urdf"))
    
    # Create versions with fixed wrists
    non_fixed_joints_fixed_wrist = [
        "joint_mobile_base_translation",
        "joint_mobile_base_rotation",
        "joint_lift",
        "joint_arm_l0",
    ]
    
    for robot in [robot_rotary, robot_prismatic]:
        print("Prepare URDF with a fixed wrist.")
        # Change any joint that should be immobile for end effector IK into a fixed joint
        for j in robot.joint_map.keys():
            if j not in non_fixed_joints_fixed_wrist:
                joint = robot.joint_map[j]
                # print('(joint name, joint type) =', (joint.name, joint.type))
                joint.type = "fixed"
    
    save_urdf_file(robot_rotary, str(output_dir / "stretch_base_rotation_ik_with_fixed_wrist.urdf"))
    save_urdf_file(robot_prismatic, str(output_dir / "stretch_base_translation_ik_with_fixed_wrist.urdf"))
    
    print()
    print("All URDF files have been generated successfully!")


if __name__ == "__main__":
    main()
