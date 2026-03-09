# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import warnings
import numpy as np
import torch
from torchvision.transforms import v2
import scipy.spatial.transform as tra
import cv2
import warnings

from stretch.agent.zmq_client import HomeRobotZmqClient
import time
from scipy.spatial.transform import Rotation as R
from stretch.motion.kinematics import HelloStretchIdx

# Lazy imports for lerobot policies
_lerobot_policies = {}
_lerobot_import_warned = False

def _lazy_import_lerobot_policy(policy_name: str):
    """Lazily import lerobot policy classes with warning if not available."""
    global _lerobot_policies, _lerobot_import_warned
    
    # Return cached import if already loaded
    if policy_name in _lerobot_policies:
        return _lerobot_policies[policy_name]
    
    try:
        if policy_name == "act":
            from lerobot.common.policies.act.modeling_act import ACTPolicy # type: ignore
            _lerobot_policies[policy_name] = ACTPolicy
            return ACTPolicy
        elif policy_name == "diffusion":
            from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy # type: ignore
            _lerobot_policies[policy_name] = DiffusionPolicy
            return DiffusionPolicy
        elif policy_name == "diffusion_depth":
            from lerobot.common.policies.diffusion_depth.modeling_diffusion import DiffusionPolicy as DPdepth # type: ignore
            _lerobot_policies[policy_name] = DPdepth
            return DPdepth
        elif policy_name == "vqbet":
            from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy # type: ignore
            _lerobot_policies[policy_name] = VQBeTPolicy
            return VQBeTPolicy
        elif policy_name == "dummy":
            from lerobot.common.policies.dummy_policy import DummyPolicy # type: ignore
            _lerobot_policies[policy_name] = DummyPolicy
            return DummyPolicy
        else:
            raise ValueError(f"Unknown policy name: {policy_name}")
    except ImportError as e:
        if not _lerobot_import_warned:
            warnings.warn(
                f"lerobot packages not available. Cannot import {policy_name} policy. "
                f"Error: {e}. Please install lerobot if you need policy functionality.",
                ImportWarning,
                stacklevel=2
            )
            _lerobot_import_warned = True
        raise ImportError(
            f"lerobot packages not available. Cannot load {policy_name} policy. "
            f"Please install lerobot: pip install lerobot"
        ) from e

SUPPORTED_POLICIES = ["act", "diffusion", "diffusion_depth", "vqbet", "dummy"]
GRIPPER_MIN=-0.3
GRIPPER_MAX=0.6

def load_policy(
    policy_name: str | None = None, policy_path: str | None = None, device: str | None = "cuda"
):
    """Loads specified policy with name and path. Current supported policies include 'act', 'diffusion'"""
    if policy_name not in SUPPORTED_POLICIES and policy_name != "dummy":
        raise NotImplementedError(
            f"{policy_name} is not a supported policy. Supported policies: {SUPPORTED_POLICIES}"
        )
    
    # Lazy import the policy class
    PolicyClass = _lazy_import_lerobot_policy(policy_name)
    
    # Load the policy
    if policy_name == "dummy":
        policy = PolicyClass(policy_path)
    else:
        policy = PolicyClass.from_pretrained(policy_path)
    
    policy.to(device)
    policy.eval()

    return policy


def prepare_state(
    raw_state: dict | None = None, teleop_mode: str | None = None, device: str | None = "cuda"
):

    # Format based on teleop mode
    # state = dt.format_state(raw_state, teleop_mode)
    state = raw_state

    # TODO This mode is only here to support old models with 7 state features. Remove when this is no longer needed
    if teleop_mode == "old_stationary_base":
        state = [
            0.0,  # Placeholder 0 for theta_vel
            state["joint_lift"],
            state["joint_arm_l0"],
            state["joint_wrist_roll"],
            state["joint_wrist_pitch"],
            state["joint_wrist_yaw"],
            state["stretch_gripper"],
        ]
    elif teleop_mode == "base_x":
        # This is the format for state space under the ROS2 backend
        state = [
            state["base_x"],
            state["base_y"],
            state["base_theta"],
            state["lift"],
            state["arm"],
            state["wrist_roll"],
            state["wrist_pitch"],
            state["wrist_yaw"],
            state["gripper_finger_right"],
        ]
    else:
        # Define explicit order for input state features
        state = [
            state["base_x"],
            state["base_x_vel"],
            state["base_y"],
            state["base_y_vel"],
            state["base_theta"],
            state["base_theta_vel"],
            state["joint_lift"],
            state["joint_arm_l0"],
            state["joint_wrist_pitch"],
            state["joint_wrist_yaw"],
            state["joint_wrist_roll"],
            state["stretch_gripper"],
        ]

    state = torch.from_numpy(np.array(state))
    state = state.to(torch.float32)
    state = state.to(device, non_blocking=True)
    state = state.unsqueeze(0)

    return state


def prepare_state_rel(observation: dict, joint_states, device: str = "cuda") -> torch.Tensor:
    # rum state xyz, quat, gripper
    ee_cam_pose = observation.ee_camera_pose
    ee_cam_pos = ee_cam_pose[:3, 3]
    ee_cam_quat = tra.Rotation.from_matrix(ee_cam_pose[:3, :3]).as_quat()
    gripper = normalize_gripper(joint_states['gripper'])
    # breakpoint()
    state =  [
        ee_cam_pos[0],
        ee_cam_pos[1],
        ee_cam_pos[2],
        ee_cam_quat[0],
        ee_cam_quat[1],
        ee_cam_quat[2],
        ee_cam_quat[3],
        gripper
    ]

    state = torch.from_numpy(np.array(state))
    state = state.to(torch.float32)
    state = state.to(device, non_blocking=True)
    state = state.unsqueeze(0)

    return state

def prepare_state_abs(observation: dict, joint_states, device: str = "cuda") -> torch.Tensor:
    # rum state xyz, quat, gripper
    ee_pose = observation.ee_pose
    ee_pos = ee_pose[:3, 3]
    ee_quat = tra.Rotation.from_matrix(ee_pose[:3, :3]).as_quat()
    gripper = normalize_gripper(joint_states['gripper'])

    state =  [
        ee_pos[0],
        ee_pos[1],
        ee_pos[2],
        ee_quat[0],
        ee_quat[1],
        ee_quat[2],
        ee_quat[3],
        gripper
    ]

    state = torch.from_numpy(np.array(state))
    state = state.to(torch.float32)
    state = state.to(device, non_blocking=True)
    state = state.unsqueeze(0)

    return state

def prepare_image(image, device):
    # :param image: (H, W, 3) in [0, 255]
    # output: (1, 3, 320, 320) in [0, 1]
    transforms = v2.Compose([v2.CenterCrop(320)])
    image = torch.from_numpy(image)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)
    image = image.to(device, non_blocking=True)
    image = transforms(image)
    image = image.unsqueeze(0)

    return image


def prepare_observations(
    raw_state: dict | None = None,
    gripper_color_image: np.ndarray | None = None,
    gripper_depth_image: np.ndarray | None = None,
    head_color_image: np.ndarray | None = None,
    head_depth_image: np.ndarray | None = None,
    teleop_mode: str | None = "stationary_base",
    device: str | None = "cuda",
):
    """Prepare state and image observations based on teleop mode and move to specified device"""

    # Prepare state
    state = prepare_state(raw_state, teleop_mode, device)

    # Prepare images
    images = [gripper_color_image, gripper_depth_image, head_color_image, head_depth_image]
    gripper_color_image, gripper_depth_image, head_color_image, head_depth_image = [
        prepare_image(x, device) for x in images
    ]

    observations = {
        "observation.state": state,
        "observation.images.gripper": gripper_color_image,
        "observation.images.head": head_color_image,
        "observation.images.gripper_depth": gripper_depth_image,
        "observation.images.head_depth": head_depth_image,
    }
    return observations


def prepare_action_dict(
    raw_actions: list | None, teleop_mode: str | None, current_base_x, action_origin
):
    """Formats actions predicted by the model into correctly labeled action_dict based on teleop mode"""
    action_dict = {}

    # TODO This mode is only here to support old models with 7 action features. Remove when this is no longer needed
    if teleop_mode == "old_stationary_base":
        action_dict["joint_mobile_base_rotate_by"] = raw_actions[0]
        action_dict["joint_lift"] = raw_actions[1]
        action_dict["joint_arm_l0"] = raw_actions[2]
        action_dict["joint_wrist_roll"] = raw_actions[3]
        action_dict["joint_wrist_pitch"] = raw_actions[4]
        action_dict["joint_wrist_yaw"] = raw_actions[5]
        action_dict["stretch_gripper"] = raw_actions[6]

    elif teleop_mode == "base_x":
        action_dict["joint_mobile_base_translation"] = raw_actions[0] - action_origin
        # Translate by is difference between predicted base_x and current base_x
        # self.current_base_x = raw_state["base_x"]
        action_dict["joint_mobile_base_translate_by"] = (
            action_dict["joint_mobile_base_translation"] - current_base_x
        )
        # action_dict["joint_mobile_base_translate_by"] = action[1]
        action_dict["joint_mobile_base_rotate_by"] = raw_actions[2]
        action_dict["joint_lift"] = raw_actions[3]
        action_dict["joint_arm_l0"] = raw_actions[4]
        action_dict["joint_wrist_roll"] = raw_actions[5]
        action_dict["joint_wrist_pitch"] = raw_actions[6]
        action_dict["joint_wrist_yaw"] = raw_actions[7]
        action_dict["stretch_gripper"] = raw_actions[8]

    return action_dict

def normalize_gripper(gripper: float) -> float:
    # normalize gripper [Gripper_MIN, Gripper_MAX] to [0, 1]
    gripper = np.clip(gripper, GRIPPER_MIN, GRIPPER_MAX)
    return (gripper - GRIPPER_MIN) / (GRIPPER_MAX - GRIPPER_MIN)

def unnormalize_gripper(gripper: float) -> float:
    # revert normalized gripper [0, 1] to [Gripper_MIN, Gripper_MAX]
    gripper = np.clip(gripper, 0, 1)
    return GRIPPER_MIN + (GRIPPER_MAX - GRIPPER_MIN) * gripper


def process_vertical_image(orig_image: np.ndarray, target_height: int, target_width: int, intrinsic: np.ndarray = None, cut_mode: str = "top"):
    """
    process the vertical image and return the new image and intrinsic matrix
    Args:
        orig_image: np.ndarray, the original image
        target_height: int, the target height of the cropped image
        target_width: int, the target width of the cropped image
        intrinsic: np.ndarray, the intrinsic matrix of the camera
        cut_mode: str, the mode to crop the image
    Returns:
        new_image: np.ndarray, the new image
        new_intrinsic: np.ndarray, the new intrinsic matrix
    """
    if orig_image.ndim == 2:
        DEPTH_MODE=True
    else:
        DEPTH_MODE=False
    orig_height, orig_width = orig_image.shape[0], orig_image.shape[1]
    assert orig_height > orig_width, "Original height must be greater than width"
    

    # 1. Determine Crop Offset
    cropped_height = orig_width
    padding = (orig_height - cropped_height) // 2
    if cut_mode == "bottom":
        y_offset = 2 * padding
    elif cut_mode == "center":
        y_offset = padding
    elif cut_mode == "top":
        y_offset = 0
    else:
        raise NotImplementedError
    
    # apply resize 
    if DEPTH_MODE:
        new_image = np.zeros((orig_width, orig_width), dtype=np.float32)
        new_image = orig_image[y_offset : y_offset + orig_width, 0 : orig_width]
        new_image_resized = cv2.resize(new_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    else:
        new_image = np.zeros((orig_width, orig_width, 3), dtype=np.uint8)
        new_image = orig_image[y_offset : y_offset + orig_width, 0 : orig_width, :]
        new_image_resized = cv2.resize(new_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        
    # 2. Update Intrinsic for Crop
    if intrinsic is not None:
        new_intrinsic = intrinsic.copy()
        # Shift principal point by the crop offset
        # cx remains same because x_offset is 0
        new_intrinsic[1, 2] = intrinsic[1, 2] - y_offset 
        
        # 3. Handle Resize
        # Note: new_image currently has shape (target_height, orig_width)
        # We are resizing it to (target_width, target_height)
        scale_x = target_width / orig_width
        scale_y = target_height / cropped_height # This is 1.0 in your current logic!
        
        # Apply scaling to the whole matrix (fx, fy, cx, cy)
        new_intrinsic[0, 0] *= scale_x
        new_intrinsic[0, 2] *= scale_x
        new_intrinsic[1, 1] *= scale_y
        new_intrinsic[1, 2] *= scale_y
    else:
        new_intrinsic = None

    return new_image_resized, new_intrinsic


def ask_for_input(prompt: str) -> bool:
    """Ask the user if the episode was successful."""
    while True:
        response = input(prompt + " (y/n): ")
        if response.lower() == "y":
            return True
        elif response.lower() == "n":
            return False
        else:
            print("Please enter 'y' or 'n'")

def dict_value_torch2numpy(data_batch: dict, exclude_keys: list = []) -> torch.Tensor:
    """
    convert all the values in the dictionary to numpy arrays
    Args:
        data_batch: dict, the data batch
    Returns:
        numpy array, the data batch
    """
    data_batch_np = {}
    for key, value in data_batch.items():
        if key in exclude_keys:
            continue
        if isinstance(value, torch.Tensor):
            if value.device != torch.device("cpu"):
                value = value.cpu()
            data_batch_np[key] = value.numpy()
        elif isinstance(value, np.ndarray):
            data_batch_np[key] = value
        elif isinstance(value, str):
            data_batch_np[key] = value
        else:
            print("[RECORD] missing key:",key, "with value:", value)
    return data_batch_np


def go_to_target_pose(
    robot: HomeRobotZmqClient,
    target_pos: np.ndarray, 
    target_quat: np.ndarray, 
    target_gripper:float, 
    max_iter_time: int = 0.1, 
    pos_err_threshold: float = 0.01, 
    rot_err_threshold: float = 2,
    gripper_err_threshold: float = 0.05,
    world_frame: bool = False,
    blocking: bool = True,
    verbose: bool = False,
):
    """
    Go to the target pose using the robot's arm and gripper.
    
    Args:
        robot: The robot client
        target_pos: Target position (3D array)
        target_quat: Target quaternion (xyzw format)
        target_gripper: Target gripper value
        max_iter: Maximum number of iterations (only used when non_blocking=True)
        pos_err_threshold: Position error threshold in meters
        rot_err_threshold: Rotation error threshold in degrees
        world_frame: Whether to use world frame
        non_blocking: If False, send command once and return True immediately.
                    If True, loop and check if target is reached.
        
    Returns:
        True if target reached (or command sent when non_blocking=False), False otherwise
    """
    # If non_blocking=False, just send the command and return
    if not blocking:
        robot.arm_to_ee_pose(
            pos = target_pos,
            quat = target_quat,
            gripper = target_gripper,
            world_frame = world_frame,
            blocking = False,
            timeout = 0.05,
            reliable = True,
        )
        return True
    
    # Otherwise, loop and check if target is reached
    target_rot = R.from_quat(target_quat)
    
    # Initialize error values in case max_iter is 0
    pos_err = float('inf')
    rot_err_deg = float('inf')
    
    start_time = time.time()
    while time.time() - start_time < max_iter_time:
        # Get current observation
        observation = robot.get_servo_observation()
        joint_states = {
            k: observation.joint[v] for k, v in HelloStretchIdx.name_to_idx.items()
        }

        ee_pose = observation.ee_pose
        current_pos = ee_pose[:3, 3]
        current_rot = R.from_matrix(ee_pose[:3, :3])
        current_gripper = joint_states['gripper']
        
        # Calculate position error
        pos_err = np.linalg.norm(current_pos - target_pos)
        
        # Calculate rotation error using quaternion distance (more robust than RPY)
        # This gives the angle between rotations in degrees
        rot_diff = target_rot.inv() * current_rot
        rot_err = np.abs(rot_diff.as_rotvec())
        rot_err_deg = np.linalg.norm(rot_err) * 180 / np.pi
        
        # Check if gripper is closed
        gripper_err = np.abs(current_gripper - target_gripper)

        # Check if target is reached
        reached = (pos_err < pos_err_threshold) and (rot_err_deg < rot_err_threshold) and (gripper_err < gripper_err_threshold) 
        if reached:
            if verbose:
                print(f"Reached target: pos_err={pos_err:.4f}m, rot_err={rot_err_deg:.2f}°, gripper_err={gripper_err:.4f}, time={time.time() - start_time:.3f}s")
            return True
        
        # Move towards target
        robot.arm_to_ee_pose(
            pos = target_pos,
            quat = target_quat,
            gripper = target_gripper,
            world_frame = world_frame,
            reliable = False,
            blocking = False,
            debug = True,
        )
        
        # Add a small delay to allow the robot to move before checking again
        # This prevents the loop from running too fast and wasting iterations
        # time.sleep(0.05)  # 50ms delay between iterations
    
    # Failed to reach target within max_iter
    print(f"Failed to reach target after {time.time() - start_time:.3f}s: pos_err={pos_err:.4f}m, rot_err={rot_err_deg:.2f}°, gripper_err={gripper_err:.4f}")
    return False


def precise_sleep(dt: float, slack_time: float=0.001, time_func=time.monotonic):
    """
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func()
    if dt > slack_time:
        time.sleep(dt - slack_time)
    t_end = t_start + dt
    while time_func() < t_end:
        pass
    return


def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return
