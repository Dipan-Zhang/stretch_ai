# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import pprint as pp
import os

import cv2
import numpy as np
import torch
import scipy.spatial.transform as tra
import open3d as o3d 
import sys

import stretch.app.dex_teleop.dex_teleop_utils as dt_utils
import stretch.utils.logger as logger
import stretch.utils.loop_stats as lt
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.core import get_parameters
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.data_tools.record_egoasis import FileDataRecorderEgoasis
import stretch.app.lfd.visualize_utils as vis_utils
import argparse
from omegaconf import OmegaConf
from easydict import EasyDict as edict 
from stretch.app.lfd.policy_utils import (
    unnormalize_gripper, normalize_gripper, ask_for_input, dict_value_torch2numpy, go_to_target_pose, precise_sleep, precise_wait
)

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
ACTION_DIM = 20
GRIPPER_GOAL_SIZE = (240, 320) # H,W
HEAD_GOAL_SIZE = (320, 240) # H,W
HOME_POS = np.array([0.0, -0.35, 0.85])
DEBUG_OFFSET = np.array([0.0,0.0,0.0])



def process_robot_state(observation: dict, joint_states) -> np.ndarray:
    # return state in format (17,) T_world_gripper, gripper_closure
    state = np.zeros(17)
    ee_pose = observation.ee_pose
    gripper = joint_states['gripper']
    gripper = normalize_gripper(gripper) # to [0, 1]
    state[:16] = ee_pose.reshape(-1)
    state[16] = gripper
    return state


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
        recording: bool = False,
        relative_motion: bool = False,
        run_policy: bool = True,
        policy_kwargs: edict = 
        {   
            "cfg": None,
            "weight_ckpt": None,
            "action_chunk_size": 15,
            # "action_meta_fpath": "/home/chenh/hanzhi_ws/egoasis3D/assets/stretchrobot_pickupbottle_relaction_meta.npz",
            # "state_meta_fpath": "/home/chenh/hanzhi_ws/egoasis3D/assets/stretchrobot_pickupbottle_state_meta.npz",
            "policy_only": True,
            "device": "cuda",
        },
        loop_rate: int = 10,
        episode_max_step: int = 300,
        add_noise_to_action: bool = False,
    ):
        self.robot = robot
        self.policy_kwargs = edict(policy_kwargs)
        self.device = self.policy_kwargs.device
        self.teleop_mode = teleop_mode
        self.depth_filter_k = depth_filter_k
        self.record_success = record_success
        self.verbose = verbose
        self.loop_rate = loop_rate
        self.add_noise_to_action = add_noise_to_action
        self.episode_max_step = episode_max_step
        # Save metadata to pass to recorder
        if logging_cfg is not None:
            instruction = logging_cfg.instruction
            logging_cfg.task_name = instruction.replace(" ", "_")
            self.metadata = {
                "backend": "ros2",
                "recording_type": "Policy evaluation",
                "user_name": logging_cfg.user_name,
                "task_name": logging_cfg.task_name,
                "env_name": logging_cfg.env_name,
                "policy_name": 'egoasis',
                "teleop_mode": self.teleop_mode,
                "policy_kwargs": self.policy_kwargs,
                "episode_max_step": self.episode_max_step,
            }
        else:
            self.metadata = {
                "backend": "ros2",
                "recording_type": "Policy evaluation",
                "user_name": "unknown",
                "task_name": logging_cfg.task_name if logging_cfg.task_name is not None else "unknown",
                "env_name": "unknown",
                "policy_name": 'egoasis',
                "teleop_mode": self.teleop_mode,
                "policy_kwargs": self.policy_kwargs,
                "episode_max_step": self.episode_max_step,
            }

        self._recording = recording
        if logging_cfg is not None and self._recording:
            self._recorder = FileDataRecorderEgoasis(
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
        self.logging_cfg = logging_cfg

        if self.dummy_inference:
            raise NotImplementedError("dummy_inference is not implemented yet")
        
        if self.relative_motion:
            print(colored('Relative motion is enabled', 'red'))


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

        # Acquire current gripper state
        current_state = process_robot_state(observation, joint_states) # (17,) T_world_gripper, gripper_closure
        obs = {
            "language_instruction": self.logging_cfg.instruction,  
            "observation.images.gripper": gripper_color_image,  # (240, 320, 3)
            "observation.depths.gripper": gripper_depth_image,  # (240, 320)
            "observation.images.head": head_color_image,  # (320, 320, 3)
            "observation.depths.head": head_depth_image,  # (320, 320)
            "HEAD_CAM_K": head_cam_K,  # (3, 3)
            "EE_CAM_K": gripper_cam_K,  # (3, 3)
            "observation.state": current_state,  # (17), T_world_gripper, gripper_closure = state[:16].reshape(4, 4), state[16:]
            "head_cam_pose": head_cam_pose,  # (4, 4)
            "ee_cam_pose": gripper_cam_pose,  # (4, 4)
        }
        return obs
    

    def wait_for_mission_start(self) -> bool:
        """
        Display standby mode with camera feed and wait for user input.
        
        Returns:
            True if mission should start (SPACEBAR pressed), False if cancelled (ESC pressed)
        """
        print("Robot is in standby mode. Press SPACEBAR to start the mission, or ESC to exit.")
        mission_started = False
        while not mission_started:
            # Get observation to show current camera feed
            obs = self.prepare_observation()
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
                return True
            
            elif key == 27:  # ESC
                print("Mission cancelled by user.")
                cv2.destroyAllWindows()
                return False
            
            # Keep robot in standby pose
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        
        return True

    def visualize_action(self, obs, outputs, visualize_3d: bool = False):
        current_state = obs["observation.state"].copy()
        head_color_resized = obs["observation.images.head"].copy()
        gripper_color_resized = obs["observation.images.gripper"].copy()
        head_cam_K_resized = obs["HEAD_CAM_K"].copy()
        gripper_cam_K_resized = obs["EE_CAM_K"].copy()
        T_base_head_cam = obs["head_cam_pose"].copy()  # (4,4)
        T_base_ee_cam = obs["ee_cam_pose"].copy()  # (4,4)
        latest_action_chunk = outputs["latest_predicted_action"].cpu().numpy().copy()

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
            cmap_name = "cool"
        else:
            cmap_name = "turbo"

        ######## DEBUG: Do 3D visualization ########
        if visualize_3d:
            pred_action_latest = outputs["latest_predicted_action"].cpu().numpy().copy()
            history_action_abs = obs["history_action"][0].cpu().numpy().copy() # [H, D]
            start_pos_world = obs["start_pos"][0].cpu().numpy().copy()[None] # [1, D]
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
            curr_pos_world = start_pos_world[0, 10:13]
            curr_rot_world = AriaUtils.rotation_6d_to_matrix(torch.from_numpy(start_pos_world[:, -6:])).numpy()[0]
            curr_pose_world = np.eye(4)
            curr_pose_world[:3, 3] = curr_pos_world
            curr_pose_world[:3, :3] = curr_rot_world
            curr_pose_world_vis = DatasetUtils.visualize_axis_o3d(curr_pose_world, [0, 1, 0], size=0.02)
            vis_action_latest = DatasetUtils.visualize_6d_trajectory(
                root_action_latest,
                size=0.01,
                cmap_name=cmap_name,
                to_mesh=True,
            )
            vis_action_history = DatasetUtils.visualize_6d_trajectory(
                root_action_history,
                size=0.01,
                cmap_name="Greens",
                to_mesh=True,
                )
            o3d.visualization.draw([pcd, vis_action_latest, vis_action_history, curr_pose_world_vis])

            cv2.imshow("gripper image", cv2.cvtColor(gripper_cam_for_vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
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

    def run(self) -> dict:
        """Take in image data and other data received by the robot and process it appropriately. Will parse the new observations, predict future actions and send the next action to the robot, and save everything to disk."""
        loop_timer = lt.LoopStats("lfd_leader_egoasis")
    
        self.robot.reset_manipulation_base_pose()
        print('reset robot manip base pose!')
        # obs = self.prepare_observation()
        # print(obs["observation.state"][:16].reshape(4, 4))
        # breakpoint()

        # # Warm up the policy
        # for i in range(3):
        #     obs = self.prepare_observation()
        #     with torch.inference_mode():
        #         self.policy.inference(obs, action_only=True)
        #         self.policy.reset()
        # self.policy.reset()
        # time_episode_start = time.time()

        try:
            # Take keyboard input to start the mission, otherwise the robot will be in standby mode
            if not self.wait_for_mission_start():
                return {}

            episode_step = 0
            while True:
                loop_timer.mark_start()

                # Get observation
                time_inference_start = time.time()
                obs = self.prepare_observation()

                action = None
                with torch.inference_mode():
                    outputs = self.policy.inference(obs, action_only=True, align_to_current_state=False) # relative cartesian pose xyz, quaternion wxyz
                    if self.add_noise_to_action:
                        # Add uniform random noise in [-1, 1] to action indices 0 and 2 (x and z positions)
                        noise = torch.rand(2, device=outputs['selected_action'].device) * 2 - 1
                        outputs['selected_action'][[0, 2]] += 0.035 * noise
                    action = outputs['selected_action'].cpu().numpy() # [ACTION_DIM]

                pos, quat, gripper, _ = action[:3], action[3:7], action[7], action[-1]
                gripper = unnormalize_gripper(gripper)

                if self._recording:
                    obs_dict_np = dict_value_torch2numpy(obs)
                    outputs_dict_np = dict_value_torch2numpy(outputs)
                    assert obs_dict_np.keys() == obs.keys(), 'observations and outputs keys do not match'
                    # assert outputs_dict_np.keys() == outputs.keys(), f'observations and outputs {outputs.keys()} shapes do not match'
                    self._recorder.add(
                        observations=obs_dict_np,
                        actions=outputs_dict_np.copy(),
                    )

                go_to_target_pose(
                    robot=self.robot,
                    target_pos=pos, 
                    target_quat=quat, 
                    target_gripper=gripper, 
                    max_iter_time=1, 
                    pos_err_threshold=0.02,  # Relaxed from 0.01 to 0.02m (2cm) for faster convergence
                    rot_err_threshold=5,  # Relaxed from 2° to 5° for faster convergence
                    gripper_err_threshold=0.1,
                    world_frame=False,
                    blocking=False,
                    )
                elapsed_time = time.time() - time_inference_start
                sleep_time = 1 / self.loop_rate - elapsed_time
                if sleep_time < 0:
                    print(f'===================> sleep time is negative: {sleep_time:.3f}s, skipping sleep')
                precise_sleep(max(sleep_time, 0)) # sleep for 0.1s to maintain loop rate
                
                if self.verbose:
                    self.visualize_action(obs, outputs, visualize_3d=False)
                
                episode_step += 1
                if episode_step >= self.episode_max_step:
                    print(f'===================> episode step {episode_step} >= episode max steps {self.episode_max_step}, stopping episode')
                    break
                    
        finally:
            if self._recording:
                if self.record_success:
                    success = ask_for_input("Was the episode successful?")
                    print("[LEADER] Writing data to disk with success = ", success)
                    self._recorder.write(success=success)
                else:
                    print("[LEADER] Writing data to disk.")
                    self._recorder.write()
            else:
                print("[LEADER] Not recording. Skipping writing data to disk.")
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
    parser.add_argument("-i", "--robot_ip", type=str, default="192.168.1.2", help="Robot IP address")
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
    parser.add_argument("--recording", action="store_true", default=False, help="Enable recording.")
    parser.add_argument("--no-record-success", action="store_true", help="Record success of episode.")
    parser.add_argument("--task_name", type=str, default="default_task")
    parser.add_argument("--user_name", type=str, default="default_user")
    parser.add_argument("--env_name", type=str, default="default_env")
    parser.add_argument("--dummy_inference", action="store_true", help="Run visualization only.")
    parser.add_argument("--depth-filter-k", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--rerun", action="store_true", help="Enable rerun server for visualization."
    )
    parser.add_argument("--show-images", action="store_true", help="Show images received by robot.")
    parser.add_argument("--relative_motion", action="store_true", help="Use relative motion.")
    parser.add_argument("--loop_rate", type=int, default=10, help="Loop rate in Hz.")
    parser.add_argument("--instruction", type=str, default="pick-and-place")
    parser.add_argument("--add_noise_to_action", action="store_true", help="Add noise to action.")
    parser.add_argument("--episode_max_step", type=int, default=300, help="Maximum number of steps per episode.")

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
    logging_cfg.task_name = args.task_name
    logging_cfg.env_name = args.env_name
    logging_cfg.user_name = args.user_name
    logging_cfg.instruction = args.instruction
    policy_cfg = edict(OmegaConf.load(args.policy_cfg))
    policy_cfg.DATA.load_tracks = False
    leader = ROS2LfdLeaderEgoasis(
        robot=robot,
        verbose=args.verbose,
        logging_cfg=logging_cfg,
        teleop_mode=args.teleop_mode,
        recording=args.recording,
        record_success=not args.no_record_success,
        policy_kwargs={
            "cfg": policy_cfg,
            "weight_ckpt": args.ckpt,
            "action_chunk_size": 15,
            "action_meta_fpath": "/home/chenh/hanzhi_ws/egoasis3D/assets/stretchrobot_PnP-Basketball_actionInworld_statistics.npz",
            "policy_only": True,
            "state_meta_fpath": None,
            # "state_meta_fpath": "/home/chenh/hanzhi_ws/egoasis3D/assets/stretchrobot_pickupbottle_state_meta.npz",
            "device": args.device,
        },
        relative_motion=args.relative_motion,
        loop_rate=args.loop_rate,
        episode_max_step=args.episode_max_step,
        add_noise_to_action=args.add_noise_to_action,
    )

    try:
        leader.run()
    except KeyboardInterrupt:
        pass

    if robot is not None:
        robot.stop()
