# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import json
import pprint as pp
import os

import cv2
import numpy as np
from sympy.logic.boolalg import true
import torch
import scipy.spatial.transform as tra
import liblzfse
import open3d as o3d 
from PIL import Image
import sys
# from lerobot.common.datasets.push_dataset_to_hub import dobbe_format_rel
import stretch.app.dex_teleop.dex_teleop_utils as dt_utils
import stretch.utils.logger as logger
import stretch.utils.loop_stats as lt
from stretch.agent.zmq_client import HomeRobotZmqClient
# from stretch.app.lfd.policy_utils import load_policy, prepare_image, prepare_state, prepare_state_rel, prepare_state_abs, process_vertical_image
from stretch.core import get_parameters
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.data_tools.record import FileDataRecorder
import stretch.app.lfd.visualize_utils as vis_utils
import argparse
from omegaconf import OmegaConf
from easydict import EasyDict as edict 
# policy HACK
sys.path.append("/home/chenh/hanzhi_ws/egoasis3D")
import utils.dataset_utils as DatasetUtils # type: ignore
from utils.viewer_utils import SceneViewer # type: ignore
import utils.aria_utils as AriaUtils # type: ignore
from policies.robot_policy_wrapper import PolicyVLAWorldModelWrapperStretchRobot # type: ignore 
import time 
from termcolor import colored
from scipy.spatial.transform import Rotation as R
PROGRESS_TH=0.85
GRIPPER_MIN=-0.3
GRIPPER_MAX=0.6
DEFAULT_FPS = 15
QUERY_HORIZON = 15
EXECUTE_HORIZON = 15
SMOOTH_WEIGHT = 0.1  # Favor recent actions
TRAJ_SCALE = 1.0
VIEWER_TYPE = "o3d"  # "viser" or "o3d"
KEYS_FOR_STREAMING = [
    # # Action Meta
    # "action_valid",
    # "action_mean",
    # "action_std",
    # "action_norm_max_bound",
    # "action_norm_min_bound",
    # "gt_action",
    # # Dynamics Meta
    # "state_valid",
    # "state_mean",
    # "state_std",
    # "state_norm_max_bound",
    # "state_norm_min_bound",
    # "gt_state",
    # "state_color",
    # Sensor Data
    "language_feature",
    "color",
    "color_gripper",
    "depth",
    "intrinsics",
    "intrinsics_gripper",
    "T_cam0_cam",
    "T_world_cam",
    "T_world_grippercam",
    "start_pos",
    # "start_state",
    # History Data
    # "history_state",
    # "history_action",
    # "history_action_rel",
    # "history_raymap",
    # "history_raymap_gripper",
    # "history_visual_feature_patch",
    # "history_visual_feature_patch_gripper",
]

# INSTRUCTION = None
# INSTRUCTION = "carry laptop"
# INSTRUCTION = "pick up laptop"
# INSTRUCTION = "open laptop"  # "open laptop"
INSTRUCTION = "pick up bottle"
ACTION_DIM = 20
GRIPPER_GOAL_SIZE = (240, 320) # H,W
HEAD_GOAL_SIZE = (320, 320) # H,W

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
    
def prepare_state_abs(observation: dict, joint_states) -> np.ndarray:
    # return state in format (17,) T_world_gripper, gripper_closure
    state = np.zeros(17)
    ee_pose = observation.ee_pose
    state[:16] = ee_pose.reshape(-1)
    state[16] = joint_states['gripper']
    return state

# def dict_value_torch2numpy(data_batch: dict) -> torch.Tensor:
#     """
#     convert all the values in the dictionary to numpy arrays
#     Args:
#         data_batch: dict, the data batch
#     Returns:
#         numpy array, the data batch
#     """
#     data_batch_np = {}
#     for key, value in data_batch.items():
#         if isinstance(value, torch.Tensor):
#             if value.device != torch.device("cpu"):
#                 value = value.cpu()
#             data_batch_np[key] = value.numpy()
#     return data_batch_np

def go_to_target_pose(
    robot: HomeRobotZmqClient, 
    target_pos: np.ndarray, 
    target_quat: np.ndarray, 
    target_gripper:float, 
    max_iter: int = 10, 
    pos_err_threshold: float = 0.01, 
    rot_err_threshold: float = 2,
    world_frame: bool = False,
    non_blocking: bool = False,
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
    # if not non_blocking:
    #     robot.arm_to_ee_pose(
    #         pos = target_pos,
    #         quat = target_quat,
    #         gripper = target_gripper,
    #         world_frame = world_frame,
    #         reliable = True,
    #         blocking = True,
    #     )
    #     return True
    
    # Otherwise, loop and check if target is reached
    target_rot = R.from_quat(target_quat)
    
    for it in range(max_iter):
        # Get current observation
        observation = robot.get_servo_observation()
        ee_pose = observation.ee_pose
        current_pos = ee_pose[:3, 3]
        current_rot = R.from_matrix(ee_pose[:3, :3])
        
        # Calculate position error
        pos_err = np.linalg.norm(current_pos - target_pos)
        
        # Calculate rotation error using quaternion distance (more robust than RPY)
        # This gives the angle between rotations in degrees
        rot_diff = target_rot.inv() * current_rot
        rot_err = np.abs(rot_diff.as_rotvec())
        rot_err_deg = np.linalg.norm(rot_err) * 180 / np.pi
        
        # Check if target is reached
        reached = (pos_err < pos_err_threshold) and (rot_err_deg < rot_err_threshold)
        if reached:
            print(f"Reached target: pos_err={pos_err:.4f}m, rot_err={rot_err_deg:.2f}°, iterations={it+1}/{max_iter}")
            return True
        
        # Move towards target
        robot.arm_to_ee_pose(
            pos = target_pos,
            quat = target_quat,
            gripper = target_gripper,
            world_frame = world_frame,
            reliable = True,
            blocking = True,
        )
    
    # Failed to reach target within max_iter
    print(f"Failed to reach target after {max_iter} iterations: pos_err={pos_err:.4f}m, rot_err={rot_err_deg:.2f}°")
    return False


class ROS2LfdLeaderEgoasis:
    def __init__(
        self,
        robot: HomeRobotZmqClient,
        policy_cfg: edict,
        policy_weight_fpath: str,
        verbose: bool = False,
        logging_cfg: edict = None,
        robot_config_path: str = "./policy_server/robot_config.yaml",
        teleop_mode: str = "base_x",
        record_success: bool = False,
        action_chunk_size=6,
        policy_only=True,
        action_meta_fpath: str = "/home/chenh/hanzhi_ws/egoasis3D/assets/stretchrobot_pickupbottle_relaction_meta.npz",
        state_meta_fpath="/home/chenh/hanzhi_ws/egoasis3D/assets/stretchrobot_pickupbottle_state_meta.npz",
        device: str = "cuda",
        depth_filter_k=None,
        disable_recording: bool = False,
        relative_motion: bool = False,
        run_policy: bool = True,
    ):
        self.robot = robot

        self.device = device
        self.teleop_mode = teleop_mode
        self.depth_filter_k = depth_filter_k
        self.record_success = record_success
        self.verbose = verbose

        # Save metadata to pass to recorder
        if INSTRUCTION is not None:
            logging_cfg.task_name = INSTRUCTION.replace(" ", "_")
        
        self.metadata = {
            "recording_type": "Policy evaluation",
            "user_name": logging_cfg.user_name,
            "task_name": logging_cfg.task_name,
            "env_name": logging_cfg.env_name,
            "policy_name": 'egoasis',
            "policy_path": policy_weight_fpath,
            "teleop_mode": self.teleop_mode,
            "backend": "ros2",
        }

        self._disable_recording = disable_recording
        self._recording = False or not self._disable_recording
        self._need_to_write = False
        self._recorder = FileDataRecorder(
            logging_cfg.data_dir, logging_cfg.task_name, logging_cfg.user_name, logging_cfg.env_name, logging_cfg.save_images, self.metadata
        )
        self.policy = PolicyVLAWorldModelWrapperStretchRobot(
            cfg=policy_cfg,
            weight_ckpt=policy_weight_fpath,
            action_chunk_size=action_chunk_size,
            policy_only=policy_only,
            action_meta_fpath=action_meta_fpath,
            state_meta_fpath=state_meta_fpath,
            online_update_robot_state=True,
            online_update_visual_state=True,
            online_update_extrinsics_state=True,
            online_update_environment_state=True,
            run_on_robot=True
        )
        dynamics_color = DatasetUtils.random_colors(4096)
        self.dynamics_color = dynamics_color
        self.policy.reset()

        self.relative_motion = relative_motion
        self._run_policy = run_policy
        self.dummy_inference = not self._run_policy
        self.current_pose = None

        if self.dummy_inference:
            raise NotImplementedError("dummy_inference is not implemented yet")
        
        if self.relative_motion:
            print(colored('Relative motion is enabled', 'red'))

    def run(self) -> dict:
        """Take in image data and other data received by the robot and process it appropriately. Will parse the new observations, predict future actions and send the next action to the robot, and save everything to disk."""
        loop_timer = lt.LoopStats("lfd_leader_egoasis")
        _t_debug = 0
    
        self.robot.reset_manipulation_base_pose()
        print('reset robot manip base pose!')

        # Go to initial pose
        obs_init = self.robot.get_servo_observation()
        curr_pos = obs_init.ee_pose[:3, 3]
        curr_quat = R.from_matrix(obs_init.ee_pose[:3, :3]).as_quat()
        self.robot.arm_to_ee_pose(
            pos = curr_pos,
            quat = curr_quat, 
            gripper = 1.0, 
            world_frame = False,
            reliable = True,
            blocking = True,
        )

        start = input("Start mission: Y/N?")
        if start.capitalize() != "Y":
            return

        try:
            while True:
                loop_timer.mark_start()

                # Get observation
                observation = self.robot.get_servo_observation()

                # Label joint states with appropriate format
                joint_states = {
                    k: observation.joint[v] for k, v in HelloStretchIdx.name_to_idx.items()
                }

                # get raw image
                gripper_color_image = observation.ee_rgb # RGB 
                gripper_depth_image = (
                    observation.ee_depth.astype(np.float32) * observation.ee_depth_scaling
                )
                head_color_image = observation.rgb
                head_depth_image = observation.depth.astype(np.float32) * observation.depth_scaling
                head_cam_K = observation.camera_K
                gripper_cam_K = observation.ee_camera_K
                gripper_cam_pose = observation.ee_camera_pose
                head_cam_pose = observation.camera_pose

                # process images to the target size
                head_color_resized, head_cam_K_resized = process_vertical_image(head_color_image, HEAD_GOAL_SIZE[0], HEAD_GOAL_SIZE[1], head_cam_K, cut_mode="top")
                head_depth_resized, _ = process_vertical_image(head_depth_image, HEAD_GOAL_SIZE[0], HEAD_GOAL_SIZE[1], head_cam_K, cut_mode="top")

                original_height, original_width = gripper_color_image.shape[:2]
                gripper_color_resized = cv2.resize(gripper_color_image, (GRIPPER_GOAL_SIZE[1], GRIPPER_GOAL_SIZE[0]))
                gripper_depth_resized = cv2.resize(gripper_depth_image,  (GRIPPER_GOAL_SIZE[1], GRIPPER_GOAL_SIZE[0]))
                gripper_cam_K_resized= gripper_cam_K.copy()
                scale_x = GRIPPER_GOAL_SIZE[1] / original_width
                scale_y = GRIPPER_GOAL_SIZE[0] / original_height
                gripper_cam_K_resized[0, 0] *= scale_x
                gripper_cam_K_resized[1, 1] *= scale_y
                gripper_cam_K_resized[0, 2] *= scale_x
                gripper_cam_K_resized[1, 2] *= scale_y

                # Acquire current gripper state
                current_state = prepare_state_abs(observation, joint_states) # (17,) T_world_gripper, gripper_closure
                obs = {
                    "language_instruction": INSTRUCTION,  
                    "observation.images.gripper": gripper_color_resized,  # (240, 320, 3)
                    "observation.depths.gripper": gripper_depth_resized,  # (240, 320)
                    "observation.images.head": head_color_resized,  # (320, 320, 3)
                    "observation.depths.head": head_depth_resized,  # (320, 320)
                    "HEAD_CAM_K": head_cam_K_resized,  # (3, 3)
                    "EE_CAM_K": gripper_cam_K_resized,  # (3, 3)
                    "observation.state": current_state,  # (17), T_world_gripper, gripper_closure = state[:16].reshape(4, 4), state[16:]
                    "head_cam_pose": head_cam_pose,  # (4, 4)
                    "ee_cam_pose": gripper_cam_pose,  # (4, 4)
                }
                

                # Send observation to polic
                action = None
                with torch.inference_mode():
                    outputs = self.policy.inference(obs, action_only=True) # relative cartesian pose xyz, quaternion wxyz
                    action = outputs['selected_action'].cpu().numpy() # [ACTION_DIM]
                    latest_action_chunk = outputs['latest_action_chunk'].cpu().numpy()

                pos = action[:3]
                quat = action[3:7]
                gripper = action[7] 
                progress = action[-1]

                if self.verbose:
                    T_base_head_cam = observation.camera_pose  # (4,4)
                    T_base_ee_cam = observation.ee_camera_pose
                    current_state_vis = current_state.copy()[:16].reshape(4, 4)
                    tra_curr_state_vis = current_state_vis[:3, 3]
                    quat_curr_state_vis = R.from_matrix(current_state_vis[:3, :3]).as_quat()
                    curr_state_vis = np.zeros(9)
                    curr_state_vis[:3] = tra_curr_state_vis
                    curr_state_vis[3:7] = quat_curr_state_vis
                    curr_state_vis[7] = current_state[16]
                    curr_state_vis = curr_state_vis.astype(np.float32)
                    head_image_for_vis = head_color_resized.copy()
                    gripper_cam_for_vis = gripper_color_resized.copy()

                    # print(f'latest_action_chunk shape: {latest_action_chunk.shape}')
                    assert len(latest_action_chunk.shape) == 2, 'latest_action_chunk should be a 2D array'
                    projected_img = vis_utils.project_action_predictions(
                        # curr_state_vis[None], 
                        latest_action_chunk,
                        # action[None],
                        T_base_head_cam.astype(np.float32),
                        head_cam_K_resized.astype(np.float32),
                        head_image_for_vis
                    ) # [320, 320]
                    projected_img_gripper = vis_utils.project_action_predictions(
                        # curr_state_vis[None], 
                        latest_action_chunk,
                        # action[None],
                        T_base_ee_cam.astype(np.float32),
                        gripper_cam_K_resized.astype(np.float32),
                        gripper_cam_for_vis
                    ) # [240, 320]
                    projected_img_gripper = projected_img_gripper[:, 40:280]

                    projected_img = cv2.resize(projected_img, (240, 240))
                    vis = np.concatenate([projected_img, projected_img_gripper], axis=1)

                    # Ensure image in imshow is uint8 BGR. projected_img is RGB.
                    cv2.imshow("projected actions", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                # remap to [0, 1] to [GRIPPER_MIN, GRIPPER_MAX]
                gripper = GRIPPER_MIN + (GRIPPER_MAX - GRIPPER_MIN) * gripper
                # print(f'[LEADER] action is {pos=}, quat={quat}, gripper={gripper}, progress={action[-1]} idx{_t_debug}')
                
                # self.robot.arm_to_ee_pose(
                #     pos = pos,
                #     quat = quat, 
                #     gripper=gripper,  # gripper
                #     world_frame=False,
                #     reliable=True,
                #     blocking=True,
                # )
                go_to_target_pose(
                    self.robot, 
                    pos, 
                    quat, 
                    gripper, 
                    max_iter=10, 
                    pos_err_threshold=0.01, 
                    rot_err_threshold=2, 
                    non_blocking=True,
                    world_frame=False)

                if progress >= PROGRESS_TH:
                    print('task succeed!')
                    break
                _t_debug += 1
                
                # # convert the inputs, output to numpy dict
                # data_batch_np = dict_value_torch2numpy(data_batch)
                # outputs_np = dict_value_torch2numpy(outputs)
                # self._recorder.add(
                #     ee_rgb=gripper_color_image,
                #     ee_depth=gripper_depth_image,
                #     ee_cam_pose=gripper_cam_pose,
                #     xyz=pos,
                #     quaternion=quat,
                # )

                if self.verbose:
                    loop_timer.mark_end()
                    loop_timer.pretty_print()

                stop = False
                PROGRESS_STOP_THRESHOLD = 0.95
                if len(action) == 9:
                    stop = action[-1] > PROGRESS_STOP_THRESHOLD


        finally:
            # Go to initial pose
            input("Open the gripper: Y/N?")
            obs = self.robot.get_servo_observation()
            curr_pos = obs.ee_pose[:3, 3]
            curr_quat = R.from_matrix(obs.ee_pose[:3, :3]).as_quat()
            self.robot.arm_to_ee_pose(
                pos = curr_pos,
                quat = curr_quat, 
                gripper = 1.0, 
                world_frame = False,
                reliable = True,
                blocking = True,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_cfg", type=str, default="/home/chenh/hanzhi_ws/egoasis_dataset/vla_stretch_pickup_bottle/config.yaml")
    parser.add_argument("--ckpt", type=str, default="/home/chenh/hanzhi_ws/egoasis_dataset/vla_stretch_pickup_bottle/iter30000.ckpt")
    parser.add_argument("-i", "--robot_ip", type=str, default="", help="Robot IP address")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-l", "--logging_cfg", type=str, default="./src/stretch/app/lfd/logging.yaml")
    parser.add_argument(
        "-s", "--save-images", action="store_true", help="Save raw images in addition to videos"
    )
    parser.add_argument("-P", "--send_port", type=int, default=4402, help="Port to send goals to.")
    parser.add_argument(
        "--teleop-mode",
        "--teleop_mode",
        type=str,
        default="base_x",
        choices=["stationary_base", "rotary_base", "base_x"],
    )
    parser.add_argument("--record-success", action="store_true", help="Record success of episode.")
    parser.add_argument("--dummy_inference", action="store_true", help="Run visualization only.")
    parser.add_argument("--depth-filter-k", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--rerun", action="store_true", help="Enable rerun server for visualization."
    )
    parser.add_argument("--show-images", action="store_true", help="Show images received by robot.")
    parser.add_argument("--relative_motion", action="store_true", help="Use relative motion.")
    args = parser.parse_args()

    # Parameters
    MANIP_MODE_CONTROLLED_JOINTS = dt_utils.get_teleop_controlled_joints(args.teleop_mode)
    parameters = get_parameters("default_planner.yaml")
    # Zmq client
    if args.dummy_inference:
        robot = None
    else:
        robot = HomeRobotZmqClient(
            robot_ip=args.robot_ip,
            send_port=args.send_port,
            parameters=parameters,
            manip_mode_controlled_joints=MANIP_MODE_CONTROLLED_JOINTS,
            enable_rerun_server=args.rerun,
        )
        robot.switch_to_manipulation_mode()
        robot.move_to_manip_posture()


    logging_cfg = edict(OmegaConf.load(args.logging_cfg))
    policy_cfg = edict(OmegaConf.load(args.policy_cfg))
    policy_cfg.DATA.load_tracks = False
    leader = ROS2LfdLeaderEgoasis(
        robot=robot,
        policy_cfg=policy_cfg,
        policy_weight_fpath=args.ckpt,
        verbose=args.verbose,
        logging_cfg=logging_cfg,
        teleop_mode=args.teleop_mode,
        record_success=args.record_success,
        device=args.device,
        relative_motion=args.relative_motion,
        run_policy=not args.dummy_inference,
    )

    try:
        leader.run()
    except KeyboardInterrupt:
        pass

    robot.stop()
