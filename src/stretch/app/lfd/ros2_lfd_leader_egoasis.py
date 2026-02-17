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
from stretch.core import get_parameters
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.data_tools.record import FileDataRecorder
import stretch.app.lfd.visualize_utils as vis_utils
import argparse
from omegaconf import OmegaConf
from easydict import EasyDict as edict 
from stretch.app.lfd.policy_utils import process_vertical_image, unnormalize_gripper, normalize_gripper, ask_for_input
# policy HACK
sys.path.append("/home/chenh/hanzhi_ws/egoasis3D")
import utils.dataset_utils as DatasetUtils # type: ignore
from utils.viewer_utils import SceneViewer # type: ignore
import utils.aria_utils as AriaUtils # type: ignore
from policies.robot_policy_wrapper import PolicyVLAWorldModelWrapperStretchRobot # type: ignore 
import time 
from termcolor import colored
from scipy.spatial.transform import Rotation as R
PROGRESS_TH=0.95

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
INSTRUCTION = "pick up pot and place in box"
ACTION_DIM = 20
GRIPPER_GOAL_SIZE = (240, 320) # H,W
HEAD_GOAL_SIZE = (320, 320) # H,W
HOME_POS = np.array([-0.025, -0.35, 0.85])

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

def process_robot_state(observation: dict, joint_states) -> np.ndarray:
    # return state in format (17,) T_world_gripper, gripper_closure
    state = np.zeros(17)
    ee_pose = observation.ee_pose
    gripper = joint_states['gripper']
    gripper = normalize_gripper(gripper) # to [0, 1]
    state[:16] = ee_pose.reshape(-1)
    state[16] = gripper
    return state

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
    return data_batch_np

class ROS2LfdLeaderEgoasis:
    def __init__(
        self,
        robot: HomeRobotZmqClient,
        verbose: bool = False,
        logging_cfg: edict = None,
        robot_config_path: str = "./policy_server/robot_config.yaml",
        teleop_mode: str = "base_x",
        record_success: bool = False,
        depth_filter_k=None,
        disable_recording: bool = False,
        relative_motion: bool = False,
        run_policy: bool = True,
        policy_kwargs: dict = 
        {   
            "cfg": None,
            "weight_ckpt": None,
            "action_chunk_size": 8,
            # "action_meta_fpath": "/home/chenh/hanzhi_ws/egoasis3D/assets/stretchrobot_pickupbottle_relaction_meta.npz",
            # "state_meta_fpath": "/home/chenh/hanzhi_ws/egoasis3D/assets/stretchrobot_pickupbottle_state_meta.npz",
            "policy_only": True,
            "device": "cuda",
        },

    ):
        self.robot = robot
        self.policy_kwargs = edict(policy_kwargs)
        self.device = self.policy_kwargs.device
        self.teleop_mode = teleop_mode
        self.depth_filter_k = depth_filter_k
        self.record_success = record_success
        self.verbose = verbose

        # Save metadata to pass to recorder
        if logging_cfg is not None:
            if INSTRUCTION is not None:
                logging_cfg.task_name = INSTRUCTION.replace(" ", "_")
            
            self.metadata = {
                "backend": "ros2",
                "recording_type": "Policy evaluation",
                "user_name": logging_cfg.user_name,
                "task_name": logging_cfg.task_name,
                "env_name": logging_cfg.env_name,
                "policy_name": 'egoasis',
                "teleop_mode": self.teleop_mode,
                "policy_kwargs": self.policy_kwargs,
            }
        else:
            self.metadata = {
                "backend": "ros2",
                "recording_type": "Policy evaluation",
                "user_name": "unknown",
                "task_name": INSTRUCTION.replace(" ", "_") if INSTRUCTION is not None else "unknown",
                "env_name": "unknown",
                "policy_name": 'egoasis',
                "teleop_mode": self.teleop_mode,
                "policy_kwargs": self.policy_kwargs,
            }

        self._disable_recording = disable_recording
        self._recording = not self._disable_recording
        self._need_to_write = False
        if logging_cfg is not None:
            self._recorder = FileDataRecorder(
                logging_cfg.data_dir, logging_cfg.task_name, logging_cfg.user_name, logging_cfg.env_name, logging_cfg.save_images, self.metadata
            )
        else:
            self._recorder = None
        self.policy = PolicyVLAWorldModelWrapperStretchRobot(
            cfg=self.policy_kwargs.cfg,
            weight_ckpt=self.policy_kwargs.weight_ckpt,
            action_chunk_size=self.policy_kwargs.action_chunk_size,
            policy_only=self.policy_kwargs.policy_only,
            action_meta_fpath=self.policy_kwargs.action_meta_fpath,
            state_meta_fpath=self.policy_kwargs.state_meta_fpath,
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


    def go_to_target_pose(
        self,
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
            self.robot.arm_to_ee_pose(
                pos = target_pos,
                quat = target_quat,
                gripper = target_gripper,
                world_frame = world_frame,
                reliable = True,
                blocking = False,
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
            observation = self.robot.get_servo_observation()
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
            self.robot.arm_to_ee_pose(
                pos = target_pos,
                quat = target_quat,
                gripper = target_gripper,
                world_frame = world_frame,
                reliable = True,
                blocking = False,
            )
            
            # Add a small delay to allow the robot to move before checking again
            # This prevents the loop from running too fast and wasting iterations
            time.sleep(0.05)  # 50ms delay between iterations
        
        # Failed to reach target within max_iter
        print(f"Failed to reach target after {time.time() - start_time:.3f}s: pos_err={pos_err:.4f}m, rot_err={rot_err_deg:.2f}°, gripper_err={gripper_err:.4f}")
        return False

    def prepare_observation(self) -> dict:
        observation = self.robot.get_servo_observation()    
        # Label joint states with appropriate format
        joint_states = {
            k: observation.joint[v] for k, v in HelloStretchIdx.name_to_idx.items()
        }

        # get raw image
        gripper_color_image = observation.ee_rgb # RGB 
        gripper_depth_image = (
            observation.ee_depth.astype(np.float32) 
        )
        head_color_image = observation.rgb
        head_depth_image = observation.depth.astype(np.float32) 
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
        current_state = process_robot_state(observation, joint_states) # (17,) T_world_gripper, gripper_closure
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
        return obs
    

    def visualize_action(self, obs, outputs, visualize_3d: bool = False):
        current_state = obs["observation.state"].copy()
        head_color_resized = obs["observation.images.head"].copy()
        gripper_color_resized = obs["observation.images.gripper"].copy()
        head_cam_K_resized = obs["HEAD_CAM_K"].copy()
        gripper_cam_K_resized = obs["EE_CAM_K"].copy()
        T_base_head_cam = obs["head_cam_pose"].copy()  # (4,4)
        T_base_ee_cam = obs["ee_cam_pose"].copy()  # (4,4)
        latest_action_chunk = outputs["latest_action_chunk"].cpu().numpy().copy()

        current_state_vis = current_state[:16].reshape(4, 4)
        tra_curr_state_vis = current_state_vis[:3, 3]
        quat_curr_state_vis = R.from_matrix(current_state_vis[:3, :3]).as_quat()
        curr_state_vis = np.zeros(9)
        curr_state_vis[:3] = tra_curr_state_vis
        curr_state_vis[3:7] = quat_curr_state_vis
        curr_state_vis[7] = current_state[16]
        curr_state_vis = curr_state_vis.astype(np.float32)
        head_image_for_vis = head_color_resized
        gripper_cam_for_vis = gripper_color_resized
        closure = latest_action_chunk[0, 7]
        if closure < 0.5:
            cmap_name = "turbo"
        else:
            cmap_name = "cool"

        ######## DEBUG: Do 3D visualization ########
        if visualize_3d:
            pred_action_latest = outputs["latest_predicted_action"].cpu().numpy().copy()
            history_action_abs = obs["history_action_abs"][0].cpu().numpy().copy() # [H, D]
            start_pos_world = obs["start_pos_world"][0].cpu().numpy().copy()[None] # [1, D]
            assert history_action_abs.shape[1] == 20
            history_action_right = history_action_abs[:, 10:]

            
            points_3d, scene_ids = DatasetUtils.backproject(obs["observation.depths.head"], 
                                                obs["HEAD_CAM_K"], 
                                                obs["observation.depths.head"] < 1.5, 
                                                NOCS_convention=False)
            T_world_head_cam = obs["head_cam_pose"].copy()
            points_world = DatasetUtils.transform_points(points_3d, T_world_head_cam)
            points_colors = obs["observation.images.head"][scene_ids[0], scene_ids[1]] / 255.0
            pcd = DatasetUtils.visualize_points(points_world, points_colors)

            root_action_history = DatasetUtils.get_root_transformation(history_action_right)
            tra_action_latest = pred_action_latest[:, :3] # [H, 3]
            quat_action_latest = pred_action_latest[:, 3:7] # [H, 4]
            rot_action_latest = R.from_quat(quat_action_latest).as_matrix() # [H, 3, 3]
            root_action_latest = np.eye(4)[None].repeat(tra_action_latest.shape[0], axis=0)
            root_action_latest[:, :3, 3] = tra_action_latest
            root_action_latest[:, :3, :3] = rot_action_latest
            curr_pos_world = DatasetUtils.visualize_sphere_o3d(start_pos_world[0, :3], [0, 1, 0], size=0.02)
            vis_action_latest = DatasetUtils.visualize_6d_trajectory(
                root_action_latest,
                size=0.01,
                cmap_name=cmap_name,
                to_mesh=True,
            )
            vis_action_history = DatasetUtils.visualize_6d_trajectory(
                root_action_history,
                size=0.01,
                cmap_name="hot",
                to_mesh=True,
                )
            o3d.visualization.draw([pcd, vis_action_latest, vis_action_history, curr_pos_world])
        ######## DEBUG: Do 3D visualization ########
        else:
            assert len(latest_action_chunk.shape) == 2, 'latest_action_chunk should be a 2D array'
            projected_img = vis_utils.project_action_predictions(
                # curr_state_vis[None], 
                latest_action_chunk,
                # action[None],
                T_base_head_cam.astype(np.float32),
                head_cam_K_resized.astype(np.float32),
                head_image_for_vis,
                cmap_name=cmap_name
            ) # [320, 320]
            projected_img_gripper = vis_utils.project_action_predictions(
                # curr_state_vis[None], 
                latest_action_chunk,
                # action[None],
                T_base_ee_cam.astype(np.float32),
                gripper_cam_K_resized.astype(np.float32),
                gripper_cam_for_vis,
                cmap_name=cmap_name
            ) # [240, 320]
            projected_img_gripper = projected_img_gripper[:, 40:280]

            projected_img = cv2.resize(projected_img, (240, 240))
            vis = np.concatenate([projected_img, projected_img_gripper], axis=1)

            # Ensure image in imshow is uint8 BGR. projected_img is RGB.
            cv2.imshow("projected actions", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


    def robot_standby(self):
        obs_init = self.robot.get_servo_observation()
        curr_pos = obs_init.ee_pose[:3, 3]
        curr_quat = R.from_matrix(obs_init.ee_pose[:3, :3]).as_quat()
        self.go_to_target_pose(
            target_pos=curr_pos, 
            target_quat=curr_quat, 
            target_gripper=0.9, 
            max_iter_time=2, 
            pos_err_threshold=0.01, 
            rot_err_threshold=1, 
            gripper_err_threshold=0.05,
            world_frame=False,
            blocking=True,
        )
        time.sleep(0.05)

    def run(self) -> dict:
        """Take in image data and other data received by the robot and process it appropriately. Will parse the new observations, predict future actions and send the next action to the robot, and save everything to disk."""
        loop_timer = lt.LoopStats("lfd_leader_egoasis")
    
        self.robot.reset_manipulation_base_pose()
        print('reset robot manip base pose!')

        # Go to initial pose
        self.robot_standby()

        # Warm up the policy
        for i in range(10):
            obs = self.prepare_observation()
            with torch.inference_mode():
                self.policy.inference(obs, action_only=True)
                self.policy.reset()
        self.policy.reset()
        time_episode_start = time.time()

        try:
            # Take keyboard input to start the mission, otherwise the robot will be in standby mode
            print("Robot is in standby mode. Press SPACEBAR to start the mission, or ESC to exit.")
            mission_started = False
            while not mission_started:
                # Get observation to show current camera feed
                obs = self.prepare_observation()
                head_image = obs["observation.images.head"]
                gripper_image = obs["observation.images.gripper"]
                
                # Create a combined view for standby mode
                gripper_image_bgr = cv2.cvtColor(gripper_image, cv2.COLOR_RGB2BGR)
                gripper_resized = cv2.resize(gripper_image_bgr, (320, 240))

                cv2.putText(gripper_resized, "STANDBY - Press SPACE to start", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                
                cv2.imshow("Standby Mode - Press SPACE to start mission", gripper_resized)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # SPACEBAR
                    self.policy.reset()
                    time.sleep(0.1)
                    print("Mission started!")
                    mission_started = True
                    cv2.destroyAllWindows()
                    time.sleep(0.1)
                    break
                
                elif key == 27:  # ESC
                    print("Mission cancelled by user.")
                    return {}
                
                # Keep robot in standby pose
                self.robot_standby()
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage

            while True:
                loop_timer.mark_start()

                # Get observation
                time_inference_start = time.time()
                obs = self.prepare_observation()

                action = None
                with torch.inference_mode():
                    outputs = self.policy.inference(obs, action_only=True, align_to_current_state=False) # relative cartesian pose xyz, quaternion wxyz
                    action = outputs['selected_action'].cpu().numpy() # [ACTION_DIM]
                # print(f'===================> inference time: {time.time() - time_start:.3f}s')
                pos, quat, gripper, progress = action[:3], action[3:7], action[7], action[-1]

                history_gripper = obs["observation.state"][16]
                action_chunk_gripper = outputs["latest_action_chunk"][:, 7].cpu().numpy().copy().astype(np.float32)
                print(f"===================> {history_gripper:.3f}=, {gripper:.3f}=, {action_chunk_gripper}= ")
                # breakpoint()
                gripper = unnormalize_gripper(gripper)

                if self.verbose:
                    self.visualize_action(obs, outputs, visualize_3d=False)
                    # loop_timer.mark_end()
                    # loop_timer.pretty_print()
                
                # if not self._disable_recording:
                #     # prepare obs and output dict
                #     obs_dict_np = dict_value_torch2numpy(obs)
                #     outputs_dict_np = dict_value_torch2numpy(outputs)
                    
                #     self._recorder.add(
                #         ee_cam_pose=obs["ee_cam_pose"].copy(),
                #         head_cam_pose=obs["head_cam_pose"].copy(),
                #         ee_rgb=obs["observation.images.gripper"].copy(),
                #         ee_depth=obs["observation.depths.gripper"].copy(),
                #         xyz=np.array([0]),
                #         quaternion=np.array([0]),
                #         gripper=0,
                #         ee_pose=np.array([0]),
                #         observations=,
                #         actions=action,
                #         head_rgb=obs["observation.images.head"],
                #         head_depth=obs["observation.depths.head"],
                #     )

                self.go_to_target_pose(
                    target_pos=pos, 
                    target_quat=quat, 
                    target_gripper=gripper, 
                    max_iter_time=1, 
                    pos_err_threshold=0.02,  # Relaxed from 0.01 to 0.02m (2cm) for faster convergence
                    rot_err_threshold=5,  # Relaxed from 2° to 5° for faster convergence
                    gripper_err_threshold=0.1,
                    world_frame=False,
                    blocking=True,
                    )
                elapsed_time = time.time() - time_inference_start
                # print(f'===================> action execution time: {elapsed_time:.3f}s')
                precise_sleep(0.1 - elapsed_time) # sleep for 0.1s to maintain 10Hz loop rate

                if progress >= PROGRESS_TH:
                    print('task succeed!')
                    break

        finally:
            # Go to initial pose
            if ask_for_input("Confirm go to home pose?"):
                obs = self.robot.get_servo_observation()
                self.robot.arm_to_ee_pose(
                    pos = HOME_POS,
                    quat = None, 
                    gripper = 1.0, 
                    world_frame = False,
                    reliable = True,
                    blocking = True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--policy_cfg", type=str, default="/home/chenh/hanzhi_ws/egoasis_dataset/vla_stretch_potpicknplace_wprogres/config.yaml")
    parser.add_argument("--ckpt", type=str, default="/home/chenh/hanzhi_ws/egoasis_dataset/vla_stretch_potpicknplace_wprogres/iter28000.ckpt")
    parser.add_argument("-i", "--robot_ip", type=str, default="192.168.1.10", help="Robot IP address")
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
        verbose=args.verbose,
        logging_cfg=logging_cfg,
        teleop_mode=args.teleop_mode,
        record_success=args.record_success,
        policy_kwargs={
            "cfg": policy_cfg,
            "weight_ckpt": args.ckpt,
            "action_chunk_size": 12,
            "policy_only": True,
            "action_meta_fpath": "/home/chenh/hanzhi_ws/egoasis3D/assets/stretchrobot_pick-and-place_relaction_meta.npz",
            "state_meta_fpath": None,
            # "state_meta_fpath": "/home/chenh/hanzhi_ws/egoasis3D/assets/stretchrobot_pickupbottle_state_meta.npz",
            "device": args.device,
        },
        relative_motion=args.relative_motion,
    )

    try:
        leader.run()
    except KeyboardInterrupt:
        pass

    if robot is not None:
        robot.stop()
