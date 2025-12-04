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

from lerobot.common.datasets.push_dataset_to_hub import dobbe_format_rel
import stretch.app.dex_teleop.dex_teleop_utils as dt_utils
import stretch.utils.logger as logger
import stretch.utils.loop_stats as lt
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.app.lfd.policy_utils import load_policy, prepare_image, prepare_state, prepare_state_rel, prepare_state_abs
from stretch.core import get_parameters
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.data_tools.record import FileDataRecorder
import stretch.app.lfd.visualize_utils as vis_utils
import liblzfse
import open3d as o3d 

class ROS2LfdLeader:
    """ROS2 version of leader for evaluating trained LfD policies with Stretch. To be used in conjunction with stretch_ros2_bridge server"""

    def __init__(
        self,
        robot: HomeRobotZmqClient,
        verbose: bool = False,
        data_dir: str = "./data",
        task_name: str = "task",
        user_name: str = "default_user",
        env_name: str = "default_env",
        force_execute: bool = False,
        save_images: bool = False,
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
    ):
        self.robot = robot

        self.save_images = save_images
        self.device = device
        self.policy_path = policy_path
        self.teleop_mode = teleop_mode
        self.depth_filter_k = depth_filter_k
        self.record_success = record_success
        self.verbose = verbose

        # Save metadata to pass to recorder
        self.metadata = {
            "recording_type": "Policy evaluation",
            "user_name": user_name,
            "task_name": task_name,
            "env_name": env_name,
            "policy_name": policy_name,
            "policy_path": policy_path,
            "teleop_mode": self.teleop_mode,
            "backend": "ros2",
        }

        self._disable_recording = disable_recording
        self._recording = False or not self._disable_recording
        self._need_to_write = False
        self._recorder = FileDataRecorder(
            data_dir, task_name, user_name, env_name, save_images, self.metadata
        )
        self.policy = load_policy(policy_name, policy_path, device)
        self.policy.reset()
        self.relative_motion = relative_motion
        self._run_policy = run_policy
        self.visualize_trajectory = not self._run_policy
        self.visualization_data_path = visualization_data_path

        if self.visualize_trajectory:
            assert self.visualization_data_path is not None, 'visualization_data_path must be provided when visualize_trajectory is enabled'
        
        if self.relative_motion:
            assert ('rel' in policy_path) or ('rum' in policy_path), 'Policy path is for relative motion, but relative motion is disabled. Please check the policy path.'

    def run(self) -> dict:
        """Take in image data and other data received by the robot and process it appropriately. Will parse the new observations, predict future actions and send the next action to the robot, and save everything to disk."""
        loop_timer = lt.LoopStats("lfd_leader_ee")
        First = True
        _t_debug = 0
        
        # Visualization mode: test inference with ground truth data
        if self.visualize_trajectory:
            if self.visualization_data_path is None:
                raise ValueError("visualization_data_path must be provided when visualize_trajectory is enabled")

            for idx in range (0, 200, 8):
                vis_utils.visualize_trajectory(self.policy, self.relative_motion, self.visualization_data_path, idx)
            breakpoint()

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
                gripper_color_image = cv2.cvtColor(observation.ee_rgb, cv2.COLOR_RGB2BGR)
                gripper_depth_image = (
                    observation.ee_depth.astype(np.float32) * observation.ee_depth_scaling
                )
                head_color_image = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)
                head_depth_image = observation.depth.astype(np.float32) * observation.depth_scaling
                print('gripper_color_image shape', gripper_depth_image.shape)
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

                    current_img = prepare_image(
                        gripper_color_image, self.device
                    )# [:, [2,1,0]] # in RGB format

                    # ### DEBUG VISUALIZE THE IMAGE
                    _img_uint8 = (current_img[0] * 255).cpu().numpy().astype(np.uint8)
                    _img_uint8 = np.transpose(_img_uint8, (1,2,0))

                    # cv2.imshow("observation images", _img_uint8)
                    # cv2.waitKey(1)

                    observations = {
                        "observation.state": current_state,
                        "observation.images.gripper": current_img,
                        "observation.images.head": prepare_image(head_color_image, self.device),
                    }

                    # Send observation to policy
                    with torch.inference_mode():
                        raw_action = self.policy.select_action(observations) # relative cartesian pose xyz, quaternion wxyz

                    action = raw_action[0].tolist() # [n_action, n_dim]
                    if self.relative_motion:
                        # Every 8 steps (new action chunk), refresh current_pose from observation
                        # This happens at the START of a new chunk, before applying the first action
                        # Following the pattern from test_actionchunk_rel2abs: when starting a chunk,
                        # we need the current absolute pose, then apply all actions in the chunk sequentially
                        if (_t_debug) % 8 == 0:
                            # Get current absolute pose from robot (this is the pose BEFORE applying the first action of the chunk)
                            current_pose = observation.ee_pose.copy()
                            print(f'idx {_t_debug}: current pose refreshed for new chunk!')
                            print(f'  Current pose position: {current_pose=}')
                        
                        # Ensure current_pose is initialized (shouldn't happen, but safety check)
                        if 'current_pose' not in locals():
                            current_pose = observation.ee_pose.copy()
                            print(f'WARNING: current_pose not initialized, using observation.ee_pose')
                        
                        # Build relative transformation matrix from action
                        # Action format: [x, y, z, qx, qy, qz, qw, gripper] (scipy uses [x, y, z, w])
                        T_rel = np.eye(4)
                        T_rel[:3, 3] = np.array(action[:3])  # Translation
                        quat_action = np.array(action[3:7])  # [qx, qy, qz, qw]
                        T_rel[:3, :3] = tra.Rotation.from_quat(quat_action).as_matrix()
                        
                        # Apply relative transformation: new_abs = current_abs @ T_rel
                        # This follows the same pattern as test: current_pose @ action
                        current_pose = current_pose @ T_rel
                        # Extract position and quaternion from resulting absolute pose
                        pos = current_pose[:3, 3]
                        quat = tra.Rotation.from_matrix(current_pose[:3, :3]).as_quat()  # Returns [x, y, z, w]
                        gripper = action[7]
                    else:
                        pos = action[:3]
                        quat = action[3:7]
                        gripper = action[7] 

                    # TEMP, remove this after adapting the dataset preparation
                    # remap to [0, 1] to [GRIPPER_MIN, GRIPPER_MAX]
                    GRIPPER_MIN=-0.3
                    GRIPPER_MAX=0.6
                    gripper = GRIPPER_MIN + (GRIPPER_MAX - GRIPPER_MIN) * gripper
        
                    print(f'[LEADER] action is {pos=}, quat={quat}, gripper={gripper}, progress={action[8]} idx{_t_debug}')
                    self.robot.arm_to_ee_pose(
                        pos = pos,
                        quat = quat, 
                        gripper=gripper,  # gripper
                        world_frame=True,
                        reliable=True,
                        blocking=True,
                    )
                    
                else:
                    # If we aren't running the policy, what do we even need to do?
                    continue  # Skip the rest of the loop
                _t_debug += 1
                First=False

                if self.verbose:
                    loop_timer.mark_end()
                    loop_timer.pretty_print()

                # Stop condition for forced execution

                stop = False
                PROGRESS_STOP_THRESHOLD = 0.95
                if len(action) == 9:
                    stop = action[8] > PROGRESS_STOP_THRESHOLD

                if self._force and stop:
                    print(f"[LEADER] Stopping policy execution")
                    self._need_to_write = True

                    if self._disable_recording:
                        self._need_to_write = False
                        break

                

        finally:
            pass

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
    parser.add_argument("-u", "--user-name", type=str, default="default_user")
    parser.add_argument("-t", "--task-name", type=str, default="default_task")
    parser.add_argument("-e", "--env-name", type=str, default="default_env")
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force execute policy right away."
    )
    parser.add_argument("-d", "--data-dir", type=str, default="./data")
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
    parser.add_argument(
        "--policy_path", type=str, required=True, help="Path to folder storing model weights"
    )
    parser.add_argument("--policy_name", type=str, required=True)

    parser.add_argument("--run_visualization", action="store_true", help="Run visualization only.")
    parser.add_argument("--depth-filter-k", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--rerun", action="store_true", help="Enable rerun server for visualization."
    )
    parser.add_argument("--show-images", action="store_true", help="Show images received by robot.")
    parser.add_argument("--relative_motion", action="store_true", help="Use relative motion.")
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
        data_dir=args.data_dir,
        user_name=args.user_name,
        task_name=args.task_name,
        env_name=args.env_name,
        force_execute=args.force,
        save_images=args.save_images,
        teleop_mode=args.teleop_mode,
        record_success=args.record_success,
        policy_name=args.policy_name,
        policy_path=args.policy_path,
        device=args.device,
        relative_motion=args.relative_motion,
        visualization_data_path=args.visualization_data_path,
        run_policy=not args.run_visualization
    )

    try:
        leader.run()
    except KeyboardInterrupt:
        pass

    robot.stop()
