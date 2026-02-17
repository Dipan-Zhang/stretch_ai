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
from stretch.app.lfd.policy_utils import load_policy, prepare_image, prepare_state, prepare_state_rel, prepare_state_abs, normalize_gripper, unnormalize_gripper, process_vertical_image, ask_for_input
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
from easydict import EasyDict as edict 
from omegaconf import OmegaConf


PROGRESS_TH=0.9
GRIPPER_GOAL_SIZE = (240, 320) # H,W
HEAD_GOAL_SIZE = (320, 240) # H,W
DEBUG_OFFSET = np.array([0,0.06,0.0])
HOME_POS = np.array([-0.025, -0.35, 0.85])

class ROS2LfdLeader:
    """ROS2 version of leader for evaluating trained LfD policies with Stretch. To be used in conjunction with stretch_ros2_bridge server"""

    def __init__(
        self,
        robot: HomeRobotZmqClient,
        verbose: bool = False,
        teleop_mode: str = "base_x",
        policy_path: str = None,
        policy_name: str = None,
        device: str = "cuda",
        depth_filter_k=None,
        relative_motion: bool = False,
        logging_cfg: edict = None,
        recording: bool = False,
        record_success: bool = True,
        automatic_reset: bool = False,
        visualize: bool = False,
        perf_debug: bool = False,

    ):
        self.robot = robot

        self.save_images = logging_cfg.save_images
        self.device = device
        self.policy_path = policy_path
        self.teleop_mode = teleop_mode
        self.depth_filter_k = depth_filter_k
        self.record_success = record_success
        self.verbose = verbose
        self.perf_debug = perf_debug
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

        self._recording = recording
        self.automatic_reset = automatic_reset
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
        # Track current pose for relative motion mode
        self.current_pose = None
        self.visualize = visualize

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
        
        # Initialize current_pose for relative motion mode
        if self.relative_motion:
            self.current_pose = obs_init.ee_pose.copy()

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
        # Build state observations in correct format
        joint_states = {
            k: observation.joint[v] for k, v in HelloStretchIdx.name_to_idx.items()
        }
        gripper_joint = joint_states['gripper']

        if self.relative_motion:
            current_state = prepare_state_rel(observation, joint_states, self.device)
        else:
            current_state = prepare_state_abs(observation, joint_states, self.device)

        # get raw image
        gripper_color_image = observation.ee_rgb # RGB 
        gripper_depth_image = observation.ee_depth.astype(np.float32)
        gripper_cam_K = observation.ee_camera_K
        gripper_cam_pose = observation.ee_camera_pose

        head_color_image = observation.rgb
        head_depth_image = observation.depth.astype(np.float32) 
        head_cam_K = observation.camera_K
        head_cam_pose = observation.camera_pose

        assert gripper_color_image.shape == (240, 320, 3)
        assert head_color_image.shape == (320, 240, 3)
        gripper_color_resized_ts = prepare_image(gripper_color_image, self.device)
        head_color_resized_ts = prepare_image(head_color_image, self.device)
        gripper_cam_K_resized = gripper_cam_K
        head_cam_K_resized = head_cam_K
    
        obs = {
            "observation.state": current_state,  # (17), T_world_gripper, gripper_closure = state[:16].reshape(4, 4), state[16:]
            "observation.images.gripper": gripper_color_resized_ts,  # (1, 3, 240, 320)
            # "observation.depths.gripper": gripper_depth_resized,  # (240, 320)
            "observation.images.head": head_color_resized_ts,  # (1, 3, 320, 320)
            # "observation.depths.head": head_depth_resized,  # (320, 320)
            "HEAD_CAM_K": head_cam_K_resized,  # (3, 3)
            "EE_CAM_K": gripper_cam_K_resized,  # (3, 3)
            "head_cam_pose": head_cam_pose,  # (4, 4)
            "ee_cam_pose": gripper_cam_pose,  # (4, 4)
            "images.gripper": gripper_color_image,
            "images.head": head_color_image,
            "depths.gripper": gripper_depth_image,
            "depths.head": head_depth_image,
            "ee_pose": observation.ee_pose,
            "joint_states": joint_states,
            "gripper": normalize_gripper(gripper_joint),
        }
        return obs

    def visualize_action(self, obs, action_chunk, visualize_3d: bool = False):
        # current_state = obs["observation.state"].copy()
        head_color_resized = obs["images.head"].copy() # (320, 320, 3)
        gripper_color_resized = obs["images.gripper"].copy() # (240, 320, 3)
        head_cam_K_resized = obs["HEAD_CAM_K"].copy()
        gripper_cam_K_resized = obs["EE_CAM_K"].copy()
        T_base_head_cam = obs["head_cam_pose"].copy()  # (4,4)
        T_base_ee_cam = obs["ee_cam_pose"].copy()  # (4,4)
        latest_action_chunk = action_chunk[:,0,:].cpu().numpy()

        # current_state_vis = current_state[:16].reshape(4, 4)
        # tra_curr_state_vis = current_state_vis[:3, 3]
        # quat_curr_state_vis = R.from_matrix(current_state_vis[:3, :3]).as_quat()
        # curr_state_vis = np.zeros(9)
        # curr_state_vis[:3] = tra_curr_state_vis
        # curr_state_vis[3:7] = quat_curr_state_vis
        # curr_state_vis[7] = current_state[16]
        # curr_state_vis = curr_state_vis.astype(np.float32)
        head_image_for_vis = head_color_resized.astype(np.uint8)
        gripper_cam_for_vis = gripper_color_resized.astype(np.uint8)
        closure = latest_action_chunk[0, 7]
        if closure < 0.5:
            cmap_name = "turbo"
        else:
            cmap_name = "cool"

        ######## DEBUG: Do 3D visualization ########
        if visualize_3d:
            # pred_action_latest = action_chunk.cpu().numpy().copy()
            # history_action_abs = obs["history_action_abs"][0].cpu().numpy().copy() # [H, D]
            # start_pos_world = obs["start_pos_world"][0].cpu().numpy().copy()[None] # [1, D]
            # assert history_action_abs.shape[1] == 20
            # history_action_right = history_action_abs[:, 10:]

            
            # points_3d, scene_ids = DatasetUtils.backproject(obs["observation.depths.head"], 
            #                                     obs["HEAD_CAM_K"], 
            #                                     obs["observation.depths.head"] < 1.5, 
            #                                     NOCS_convention=False)
            # T_world_head_cam = obs["head_cam_pose"].copy()
            # points_world = DatasetUtils.transform_points(points_3d, T_world_head_cam)
            # points_colors = obs["observation.images.head"][scene_ids[0], scene_ids[1]] / 255.0
            # pcd = DatasetUtils.visualize_points(points_world, points_colors)

            # root_action_history = DatasetUtils.get_root_transformation(history_action_right)
            # tra_action_latest = pred_action_latest[:, :3] # [H, 3]
            # quat_action_latest = pred_action_latest[:, 3:7] # [H, 4]
            # rot_action_latest = R.from_quat(quat_action_latest).as_matrix() # [H, 3, 3]
            # root_action_latest = np.eye(4)[None].repeat(tra_action_latest.shape[0], axis=0)
            # root_action_latest[:, :3, 3] = tra_action_latest
            # root_action_latest[:, :3, :3] = rot_action_latest
            # curr_pos_world = DatasetUtils.visualize_sphere_o3d(start_pos_world[0, :3], [0, 1, 0], size=0.02)
            # vis_action_latest = DatasetUtils.visualize_6d_trajectory(
            #     root_action_latest,
            #     size=0.01,
            #     cmap_name=cmap_name,
            #     to_mesh=True,
            # )
            # vis_action_history = DatasetUtils.visualize_6d_trajectory(
            #     root_action_history,
            #     size=0.01,
            #     cmap_name="hot",
            #     to_mesh=True,
            #     )
            # o3d.visualization.draw([pcd, vis_action_latest, vis_action_history, curr_pos_world])
            pass
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
            ) # [320, 320]
            projected_img_gripper = cv2.resize(projected_img_gripper, (240, 240))

            projected_img = cv2.resize(projected_img, (240, 240))
            vis = np.concatenate([projected_img, projected_img_gripper], axis=1)

            # Ensure image in imshow is uint8 BGR. projected_img is RGB.
            cv2.imshow("projected actions", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


    def run(self) -> dict:
        """Take in image data and other data received by the robot and process it appropriately. Will parse the new observations, predict future actions and send the next action to the robot, and save everything to disk."""
        loop_timer = lt.LoopStats("lfd_leader_ee")
        self.robot.reset_manipulation_base_pose()

        self.robot_standby()

        if not ask_for_input("Start mission? "):
            return

        if self.perf_debug:
            perf_last_print = time.perf_counter()
            perf_loop_count = 0
            perf_new_obs_count = 0
            perf_last_obs_id = None
            perf_last_servo_seq = None
        try:
            while True:
                loop_timer.mark_start()

                action = None
                observations = self.prepare_observation()
                servo_seq, _, _ = self.robot.get_servo_stats()
                if self.perf_debug:
                    perf_loop_count += 1
                    if perf_last_obs_id is None or servo_seq != perf_last_obs_id:
                        perf_new_obs_count += 1
                        perf_last_obs_id = servo_seq

                # Send observation to polic
                time_before_inference = time.time()
                with torch.inference_mode():
                    raw_action, full_actions = self.policy.select_action(observations, return_full_actions=True) # relative cartesian pose xyz, quaternion wxyz

                time_after_inference = time.time()
                if self.perf_debug:
                    print(f'inference time: {time_after_inference - time_before_inference:.3f}s')

                action = raw_action[0].tolist() # [n_action, n_dim]
                if self.relative_motion:
                    # Initialize current_pose from observation if not already set (safety check)
                    if self.current_pose is None:
                        self.current_pose = observations["ee_pose"].copy()
                    
                    # Apply relative transformation: new_abs = current_abs @ T_rel
                    T_rel = np.eye(4)
                    T_rel[:3, 3] = np.array(action[:3])  # Translation
                    quat_action = np.array(action[3:7])  # [qx, qy, qz, qw]
                    T_rel[:3, :3] = tra.Rotation.from_quat(quat_action).as_matrix()
                    
                    self.current_pose = self.current_pose @ T_rel
                    pos = self.current_pose[:3, 3]
                    quat = tra.Rotation.from_matrix(self.current_pose[:3, :3]).as_quat()  # Returns [x, y, z, w]
                    gripper = unnormalize_gripper(action[7]) 
                else:
                    pos = action[:3]
                    quat = action[3:7]
                    gripper = unnormalize_gripper(action[7])
                    # ! DEBUG ONLY, remove this after installing new arm joint
                    pos += DEBUG_OFFSET 

                if self.visualize:
                    self.visualize_action(observations, full_actions, visualize_3d=False)
    
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
                
                if self._recording:
                    # Record episode if enabled
                    observation_dict = {
                        "joint_states": observations["joint_states"],
                        "ee_pose": observations["ee_pose"].tolist(),
                        "gripper": observations["gripper"].tolist()
                    }
                    ee_goal_pose = np.eye(4)
                    ee_goal_pose[:3, 3] = action[:3]
                    ee_goal_pose[:3, :3] = tra.Rotation.from_quat(action[3:7]).as_matrix()
                    action_dict = {
                        "joint_states_goal": None,
                        "ee_goal_pose": ee_goal_pose.tolist(), # xyz, quaternion
                        "gripper_goal": action[7], # 0,1
                    }
                    self._recorder.add(
                        ee_rgb=observations["images.gripper"],
                        ee_depth=observations["depths.gripper"],
                        ee_cam_pose=observations["ee_cam_pose"],
                        ee_cam_K=observations["EE_CAM_K"],
                        xyz=np.array([0]),
                        quaternion=np.array([0]),
                        gripper=action[7],
                        ee_pose=observations["ee_pose"],
                        ee_goal_pose=np.array(action[:7]),
                        observations=observation_dict,
                        actions=action_dict, # joint_goal_configuration, ee_goal_pose, gripper_goal
                        head_rgb=observations["images.head"],
                        head_depth=observations["depths.head"],
                        head_cam_pose=observations["head_cam_pose"],
                        head_cam_K=observations["HEAD_CAM_K"],
                    )

                if self.verbose:
                    loop_timer.mark_end()
                    loop_timer.pretty_print()
                    
                stop = False
                if action[8] >= PROGRESS_TH:
                    print('task succeed!')
                    stop = True

                if self.perf_debug:
                    now = time.perf_counter()
                    dt = now - perf_last_print
                    if dt >= 2.0:
                        loop_rate = perf_loop_count / dt
                        new_obs_rate = perf_new_obs_count / dt
                        servo_seq, _last_time, servo_age = self.robot.get_servo_stats()
                        if perf_last_servo_seq is None:
                            servo_rate = None
                        else:
                            servo_rate = (servo_seq - perf_last_servo_seq) / dt
                        perf_last_servo_seq = servo_seq
                        servo_age_ms = None if servo_age is None else servo_age * 1000.0
                        servo_age_str = "N/A" if servo_age_ms is None else f"{servo_age_ms:.1f} ms"
                        if servo_rate is None:
                            print(
                                f"[PERF] loop={loop_rate:.2f} Hz, new_obs={new_obs_rate:.2f} Hz, servo_age={servo_age_str}"
                            )
                        else:
                            print(
                                f"[PERF] loop={loop_rate:.2f} Hz, new_obs={new_obs_rate:.2f} Hz, servo={servo_rate:.2f} Hz, servo_age={servo_age_str}"
                            )
                        perf_last_print = now
                        perf_loop_count = 0
                        perf_new_obs_count = 0


                if stop:
                    if self.record_success:
                        success = ask_for_input("Was the episode successful?")
                        print("[LEADER] Writing data to disk with success = ", success)
                        self._recorder.write(success=success)
                    else:
                        print("[LEADER] Writing data to disk.")
                        self._recorder.write()
                    
                    # Reset current_pose for relative motion mode after writing
                    if self.relative_motion:
                        self.current_pose = None
                    
                    if self.automatic_reset:
                        self.policy.reset()
                        self.robot.arm_to_ee_pose(
                            pos=HOME_POS,
                            quat=None, 
                            gripper=0.8, 
                            world_frame=False,
                            reliable=True,
                            blocking=True,
                        )
                        time.sleep(3.0)
                        if ask_for_input("Confirm reset position and restart mission?"):
                            self.robot_standby()
                            self.robot.reset_manipulation_base_pose()
                            # current_pose will be reinitialized in robot_standby
                            continue
                        else:
                            break
                    else:
                        break
                
        finally:
            # Go to initial pose, open the gripper
            if ask_for_input("Open the gripper?"):
                self.robot.arm_to_ee_pose(
                    pos=HOME_POS,
                    quat=None, 
                    gripper=0.8, 
                    world_frame=False,
                    reliable=True,
                    blocking=True,
                )

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
    parser.add_argument("--relative_motion", action="store_true", help="Use relative motion.")
    parser.add_argument(
        "--policy_path", type=str, required=True, help="Path to folder storing model weights"
    )
    parser.add_argument("--depth-filter-k", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--rerun", action="store_true", help="Enable rerun server for visualization."
    )
    parser.add_argument("--recording", action="store_true", default=False, help="Enable recording.")
    parser.add_argument("--task_name", type=str, default="default_task")
    parser.add_argument("--user_name", type=str, default="default_user")
    parser.add_argument("--env_name", type=str, default="default_env")
    parser.add_argument("--record-success", action="store_true", help="Record success of episode.")
    parser.add_argument("--automatic_reset", action="store_true", default=False, help="Automatic reset position and restart mission.")
    parser.add_argument("--visualize", action="store_true", help="Use relative motion.")
    parser.add_argument("--perf_debug", action="store_true", help="Enable performance debugging.")
    args = parser.parse_args()

    # Parameters
    MANIP_MODE_CONTROLLED_JOINTS = dt_utils.get_teleop_controlled_joints(args.teleop_mode)
    parameters = get_parameters("default_planner.yaml")

    # override logging config with command line arguments
    logging_cfg = edict(OmegaConf.load(args.logging_cfg))
    logging_cfg.env_name = args.env_name
    logging_cfg.task_name = args.task_name
    logging_cfg.user_name = args.user_name

    # Zmq client
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
        recording=args.recording,
        logging_cfg=logging_cfg,
        teleop_mode=args.teleop_mode,
        policy_name=args.policy_name,
        policy_path=args.policy_path,
        device=args.device,
        relative_motion=args.relative_motion,
        automatic_reset=args.automatic_reset,
        visualize=args.visualize,
        perf_debug=args.perf_debug
    )

    try:
        leader.run()
    except KeyboardInterrupt:
        pass

    robot.stop()
