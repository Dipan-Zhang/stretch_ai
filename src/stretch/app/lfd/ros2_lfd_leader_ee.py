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

from lerobot.common.datasets.push_dataset_to_hub import dobbe_format_rel # type: ignore
import stretch.app.dex_teleop.dex_teleop_utils as dt_utils
import stretch.utils.logger as logger
import stretch.utils.loop_stats as lt
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.data_tools.record import FileDataRecorder
import stretch.app.lfd.visualize_utils as vis_utils
from stretch.app.lfd.policy_utils import load_policy, prepare_image, prepare_state, prepare_state_rel, prepare_state_abs
from lerobot.common.datasets.push_dataset_to_hub import dobbe_format_rel
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
from easydict import EasyDict as edict 
from omegaconf import OmegaConf


PROGRESS_TH=0.95
GRIPPER_MIN=-0.3
GRIPPER_MAX=0.6




class ROS2LfdLeader:
    """ROS2 version of leader for evaluating trained LfD policies with Stretch. To be used in conjunction with stretch_ros2_bridge server"""

    def __init__(
        self,
        robot: HomeRobotZmqClient,
        verbose: bool = False,
        logging_cfg: edict = None,
        teleop_mode: str = "base_x",
        record_success: bool = False,
        policy_path: str = None,
        policy_name: str = None,
        device: str = "cuda",
        depth_filter_k=None,
        disable_recording: bool = False,
        relative_motion: bool = False,
        run_policy: bool = True,
        visualization_data_path: str = None,
        visualize_action: bool = False
    ):
        self.robot = robot

        self.save_images = logging_cfg.save_images
        self.device = device
        self.policy_path = policy_path
        self.teleop_mode = teleop_mode
        self.depth_filter_k = depth_filter_k
        self.record_success = record_success
        self.verbose = verbose

        # Save metadata to pass to recorder
        self.metadata = {
            "recording_type": "Policy evaluation",
            "user_name": logging_cfg.user_name,
            "task_name": logging_cfg.task_name,
            "env_name": logging_cfg.env_name,
            "policy_name": policy_name,
            "policy_path": policy_path,
            "teleop_mode": self.teleop_mode,
            "backend": "ros2",
        }

        self._disable_recording = disable_recording
        self._recording = False or not self._disable_recording
        self._need_to_write = False
        self._recorder = FileDataRecorder(
            logging_cfg.data_dir, logging_cfg.task_name, logging_cfg.user_name, logging_cfg.env_name, logging_cfg.save_images, self.metadata
        )
        self.policy = load_policy(policy_name, policy_path, device)
        self.policy.reset()
        if policy_name == "dummy":
            self.policy.set_parameters(param_dict={
                "chunk_size": 8,
                "action_type": "real" # real or fake
            })
        self.relative_motion = relative_motion
        self._run_policy = run_policy
        self.visualize_trajectory = not self._run_policy
        self.visualization_data_path = visualization_data_path
        # Track current pose for relative motion mode
        self.current_pose = None
        self.visualize_action = visualize_action

        if self.visualize_trajectory:
            assert self.visualization_data_path is not None, 'visualization_data_path must be provided when visualize_trajectory is enabled'
        
        if self.relative_motion:
            assert ('rel' in policy_path) or ('rum' in policy_path), 'Policy path is for relative motion, but relative motion is disabled. Please check the policy path.'

    def robot_standby(self):
        "run few empty actions to let the robot stand by"
        obs_init = self.robot.get_servo_observation()
        curr_pos = obs_init.ee_pose[:3, 3]
        curr_quat = R.from_matrix(obs_init.ee_pose[:3, :3]).as_quat()
        self.go_to_target_pose(
            target_pos=curr_pos, 
            target_quat=curr_quat, 
            target_gripper=0.9, 
            max_iter_time=0.1, 
            pos_err_threshold=0.01, 
            rot_err_threshold=1, 
            gripper_err_threshold=0.05,
            world_frame=False,
            blocking=True,
        )
        time.sleep(0.05)

    

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

    def run(self) -> dict:
        """Take in image data and other data received by the robot and process it appropriately. Will parse the new observations, predict future actions and send the next action to the robot, and save everything to disk."""
        loop_timer = lt.LoopStats("lfd_leader_ee")
        self.robot.reset_manipulation_base_pose()

        self.robot_standby()
        # Visualization mode: test inference with ground truth data
        # if self.visualize_trajectory:
        #     if self.visualization_data_path is None:
        #         raise ValueError("visualization_data_path must be provided when visualize_trajectory is enabled")

        #     for idx in range (0, 200, 8):
        #         vis_utils.visualize_trajectory(self.policy, self.relative_motion, self.visualization_data_path, idx)

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

                # Process images
                gripper_color_image = observation.ee_rgb # RGB 
                gripper_depth_image = (
                    observation.ee_depth.astype(np.float32) * observation.ee_depth_scaling
                )
                head_color_image = observation.rgb
                head_depth_image = observation.depth.astype(np.float32) * observation.depth_scaling
                print('gripper_color_image shape', gripper_depth_image.shape)
                gripper_color_image_resized = cv2.resize(gripper_color_image, (320, 240))
                # Clip and normalize depth
                gripper_depth_image = dobbe_format_rel.clip_and_normalize_depth(
                    gripper_depth_image, self.depth_filter_k
                )
                head_depth_image = dobbe_format_rel.clip_and_normalize_depth(
                    head_depth_image, self.depth_filter_k
                )

                action = None
                if self._run_policy:
                    # Build state observations in correct format
                    if self.relative_motion:
                        current_state = prepare_state_rel(observation, joint_states, self.device)
                    else:
                        current_state = prepare_state_abs(observation, joint_states, self.device)

                    print('current pose', current_state[:3])

                    current_img = prepare_image(
                        gripper_color_image_resized, self.device
                    )# [:, [2,1,0]] # in RGB format

                    # DEBUG preprocess head image:
                    head_image_PIL = Image.fromarray(head_color_image)
                    original_height, original_width = head_color_image.shape[:2] # head: (1280, 720)
                    goal_width, goal_height = 320, 240
                    goal_ratio = goal_width / goal_height  # 320/240 = 4/3 = 1.333
                    
                    # Keep full height, crop width from center to match 4:3 ratio
                    crop_height = original_height
                    crop_width = int(crop_height * goal_ratio)  # 720 * 1.333 = 960
                    left = (original_width - crop_width) // 2
                    top = 0
                    
                    head_image_cropped = head_image_PIL.crop((left, top, left + crop_width, top + crop_height))
                    head_image_resized = head_image_cropped.resize((goal_width, goal_height), Image.Resampling.LANCZOS)
                    head_image_resized = np.array(head_image_resized)
                    current_head_image = prepare_image(
                        head_image_resized, self.device
                    )
                    print(f'head_image_resized shape {head_image_resized.shape}')
    

                    observations = {
                        "observation.state": current_state,
                        "observation.images.gripper": current_img,
                        "observation.images.head": current_head_image,
                    }

                    # Send observation to polic
                    with torch.inference_mode():
                        raw_action, full_actions = self.policy.select_action(observations, return_full_actions=True) # relative cartesian pose xyz, quaternion wxyz

                    action = raw_action[0].tolist() # [n_action, n_dim]
                    if self.relative_motion:
                        # Every 8 steps (new action chunk), refresh current_pose from observation
                        # This happens at the START of a new chunk, before applying the first action
                        # Following the pattern from test_actionchunk_rel2abs: when starting a chunk,
                        # we need the current absolute pose, then apply all actions in the chunk sequentially
                        # if (_t_debug) % 8 == 0:
                        #     # Get current absolute pose from robot (this is the pose BEFORE applying the first action of the chunk)
                        #     self.current_pose = observation.ee_pose.copy()
                        #     print(f'idx {_t_debug}: current pose refreshed for new chunk!')
                        #     print(f'  Current pose position: {self.current_pose[:3, 3]}')
                        
                        # Build relative transformation matrix from action
                        T_rel = np.eye(4)
                        T_rel[:3, 3] = np.array(action[:3])  # Translation
                        quat_action = np.array(action[3:7])  # [qx, qy, qz, qw]
                        T_rel[:3, :3] = tra.Rotation.from_quat(quat_action).as_matrix()
                        
                        # Apply relative transformation: new_abs = current_abs @ T_rel
                        # This matches the visualization code: current_pose = current_pose @ T_rel
                        self.current_pose = self.current_pose @ T_rel
                        # Extract position and quaternion from resulting absolute pose
                        pos = self.current_pose[:3, 3]
                        quat = tra.Rotation.from_matrix(self.current_pose[:3, :3]).as_quat()  # Returns [x, y, z, w]
                        gripper = action[7]
                    else:
                        pos = action[:3]
                        quat = action[3:7]
                        gripper = action[7] 

                    if self.visualize_action:
                        # project the action to the head image
                        # action should be an (N, 8) array (N = 1)
                        T_base_head_cam = observation.camera_pose  # (4,4)
                        head_cam_K = observation.camera_K  # (3,3)
                        head_image_for_vis = np.array(head_image_PIL)

                        actions = full_actions[:,0,:].cpu().numpy() # T,1,D -> T,D
                        # print(actions.shape)

                        projected_img = vis_utils.project_action_predictions(
                            actions, 
                            T_base_head_cam.astype(np.float32),
                            head_cam_K.astype(np.float32),
                            head_image_for_vis
                        )

                        # Ensure image in imshow is uint8 BGR. projected_img is RGB.
                        img_bgr = cv2.cvtColor(projected_img, cv2.COLOR_RGB2BGR)
                        cv2.imshow("projected actions", img_bgr)
                        cv2.waitKey(1)
                        # self.visualize_action(observations, full_actions, visualize_3d=False)

                    # TEMP, remove this after adapting the dataset preparation
                    # remap to [0, 1] to [GRIPPER_MIN, GRIPPER_MAX]

                    gripper = GRIPPER_MIN + (GRIPPER_MAX - GRIPPER_MIN) * gripper
        
                    print(f'[LEADER] action is {pos=}, quat={quat}, gripper={gripper}, progress={action[8]}')
                    self.go_to_target_pose(
                        pos, 
                        quat, 
                        gripper, 
                        max_iter_time=10, 
                        pos_err_threshold=0.01, 
                        rot_err_threshold=2, 
                        world_frame=False,
                        blocking=False)
                        
                    if action[8] >= PROGRESS_TH:
                        print('task succeed!')
                        break
                else:
                    # If we aren't running the policy, what do we even need to do?
                    continue  # Skip the rest of the loop

                if self.verbose:
                    loop_timer.mark_end()
                    loop_timer.pretty_print()
                
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

def load_gt_traj(file_path: str, matrix=False) -> list[np.ndarray]:
    with open(file_path, 'r') as f:
        gt_dict = json.load(f)
    gt_actions = []
    for frame_idx in gt_dict:
        if not matrix:
            xyz = np.array(gt_dict[str(frame_idx)]['xyz'])
            quat = np.array(gt_dict[str(frame_idx)]['quats'])
            gripper = gt_dict[str(frame_idx)]['gripper']

            gt_actions.append([xyz[0], xyz[1], xyz[2],quat[0],quat[1], quat[2],quat[3], gripper, 0.0])
        else:

            relative_pose = np.eye(4)
            relative_pose[:3,3] = xyz
            relative_pose[:3,:3] = tra.Rotation.from_quat(quat).as_matrix()
            
            gt_actions.append(relative_pose)

    
    init_pose_abs = np.array(gt_dict[str(0)]['xyz_abs'])
    
    return np.array(gt_actions), init_pose_abs

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--robot_ip", type=str, default="", help="Robot IP address")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-l", "--logging_cfg", type=str, default="./src/stretch/app/lfd/logging.yaml")
    parser.add_argument(
        "-s", "--save-images", action="store_true", help="Save raw images in addition to videos"
    )
    parser.add_argument("--policy_name", type=str, required=True)
    parser.add_argument("-P", "--send_port", type=int, default=4402, help="Port to send goals to.")
    parser.add_argument(
        "--teleop-mode",
        "--teleop_mode",
        type=str,
        default="base_x",
        choices=["stationary_base", "rotary_base", "base_x"],
    )
    parser.add_argument("--record-success", action="store_true", help="Record success of episode.")
    parser.add_argument(
        "--policy_path", type=str, required=True, help="Path to folder storing model weights"
    )

    parser.add_argument("--run_visualization", action="store_true", help="Run visualization only.")
    parser.add_argument("--depth-filter-k", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--rerun", action="store_true", help="Enable rerun server for visualization."
    )
    parser.add_argument("--show-images", action="store_true", help="Show images received by robot.")
    parser.add_argument("--relative_motion", action="store_true", help="Use relative motion.")
    parser.add_argument("--visualize_action", action="store_true", help="Use relative motion.")
    parser.add_argument(
        "--visualization_data_path",
        type=str,
        default=None,
        help="Path to data directory for visualization mode (should contain compressed_gripper_images/ and labels.json)."
    )
    args = parser.parse_args()

    # Parameters
    MANIP_MODE_CONTROLLED_JOINTS = dt_utils.get_teleop_controlled_joints(args.teleop_mode)
    parameters = get_parameters("default_planner.yaml")

    logging_cfg = edict(OmegaConf.load(args.logging_cfg))

    # Zmq client
    if args.run_visualization:
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

    leader = ROS2LfdLeader(
        robot=robot,
        verbose=args.verbose,
        logging_cfg=logging_cfg,
        teleop_mode=args.teleop_mode,
        record_success=args.record_success,
        policy_name=args.policy_name,
        policy_path=args.policy_path,
        device=args.device,
        relative_motion=args.relative_motion,
        visualization_data_path=args.visualization_data_path,
        run_policy=not args.run_visualization,
        visualize_action=args.visualize_action
    )

    try:
        leader.run()
    except KeyboardInterrupt:
        pass

    robot.stop()
