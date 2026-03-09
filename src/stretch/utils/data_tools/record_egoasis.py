# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import datetime
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import liblzfse
import numpy as np
from tqdm import tqdm
import scipy.spatial.transform as tra
import stretch.utils.git_tools as git_tools

logger = logging.getLogger(__name__)

COMPLETION_FILENAME = "rgb_rel_videos_exported.txt"
IMG_COMPLETION_FILENAME = "completed.txt"
ABANDONED_FILENAME = "abandoned.txt"

RGB_VIDEO_H264_NAME = "gripper_compressed_video_h264.mp4"
HEAD_RGB_VIDEO_H264_NAME = "head_compressed_video_h264.mp4"

DEPTH_FOLDER_NAME = "compressed_gripper_depths"
RGB_FOLDER_NAME = "compressed_gripper_images"
HEAD_DEPTH_FOLDER_NAME = "compressed_head_depths"
HEAD_RGB_FOLDER_NAME = "compressed_head_images"

COMPLETED_DEPTH_FILENAME = "compressed_np_gripper_depth_float32.bin"
COMPLETED_HEAD_DEPTH_FILENAME = "compressed_np_head_depth_float32.bin"

OBS_DICT_FOLDER_NAME = "obs_dicts"
OUTPUT_DICT_FOLDER_NAME = "output_dicts"
from stretch.utils.data_tools.record import FileDataRecorder


def make_json_serializable(obj):
    """Recursively convert non-JSON-serializable objects to serializable formats.
    
    Handles:
    - numpy arrays -> lists
    - numpy scalars -> Python scalars
    - edict (EasyDict) -> dict
    - OmegaConf -> dict
    - Other custom objects -> dict or string
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        # Handle edict (EasyDict) - convert to regular dict first
        try:
            # Check if it's an edict-like object
            if hasattr(obj, '__dict__') and not isinstance(obj, dict):
                obj = dict(obj)
        except:
            pass
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # For other types (like OmegaConf, custom objects), try to convert to dict
        try:
            # Try OmegaConf conversion - OmegaConf objects have _content attribute
            if hasattr(obj, '_content'):
                return make_json_serializable(dict(obj))
            # Try general dict conversion for objects with __dict__
            if hasattr(obj, '__dict__'):
                return make_json_serializable(obj.__dict__)
            # Try to convert edict-like objects
            if hasattr(obj, 'keys') and hasattr(obj, '__getitem__'):
                return make_json_serializable(dict(obj))
        except Exception:
            pass
        # Last resort: convert to string
        return str(obj)


class FileDataRecorderEgoasis(FileDataRecorder):
    """A class for writing out data to files for use in learning from demonstration. This one will create a folder structure with images and a text file containing position information."""

    def __init__(self, *args, **kwargs):
        """Initialize the recorder.

        Args:
            datadir: The directory to save the data in.
            task: The name of the task.
            user: The name of the user.
            env: The name of the environment.
            fps: The fps to write videos at
        """
        super().__init__(*args, **kwargs)

    def reset(self):
        """Clear the data stored in the recorder."""
        self.observations_list = []
        self.outputs_list = []
        self.rgbs = []
        self.depths = []
        self.head_rgbs = []
        self.head_depths = []
        self.data_dicts = {}
        self.step = 0

    def add(
        self,
        observations: Dict[str, float],
        actions: Dict[str, float],
    ):
        """Add data to the recorder."""
        self.observations_list.append(observations)
        self.outputs_list.append(actions)
        self.step += 1

    def process_data(self):
        # extract the rgb, depth from the observations

        for step_idx, (observation, output) in enumerate(zip(self.observations_list, self.outputs_list)):
            rgb = observation["observation.images.gripper"]
            head_rgb = observation["observation.images.head"]
            depth = observation["observation.depths.gripper"]
            head_depth = observation["observation.depths.head"]
            self.rgbs.append(rgb)
            self.depths.append(depth)
            self.head_rgbs.append(head_rgb)
            self.head_depths.append(head_depth)
            
            observation_dict = {
                "joint_states": None,
                "ee_pose": observation["observation.state"][:16].reshape(4, 4).tolist(),
                "gripper": observation["observation.state"][16].tolist(),
            }
            ee_goal_pose = np.eye(4)
            ee_goal_pose[:3, 3] = output["selected_action"][:3]
            ee_goal_pose[:3, :3] = tra.Rotation.from_quat(output["selected_action"][3:7]).as_matrix()
            action_dict = {
                "joint_states_goal": None,
                "ee_goal_pose": ee_goal_pose.tolist(), # xyz, quaternion
                "gripper_goal": output["selected_action"][7].tolist(), # 0,1
                "last_action_chunk": output["latest_action_chunk"].tolist(),
            }
            self.data_dicts[step_idx] = {
                "step": step_idx,
                "ee_pose": observation["observation.state"][:16].reshape(4, 4).tolist(),
                "gripper": observation["observation.state"][16].tolist(),
                "head_cam_pose": observation["head_cam_pose"].tolist(),
                "ee_cam_pose": observation["ee_cam_pose"].tolist(),
                "ee_cam_K": observation["EE_CAM_K"].tolist(),
                "head_cam_K": observation["HEAD_CAM_K"].tolist(),
                "observations": observation_dict,
                "actions": action_dict,
            }


    def write_recorded_dict(self, episode_dir):
        """Write out the recorded dictionaries to folders in different npy files."""
        # Create directories if they don't exist
        obs_dir = episode_dir / OBS_DICT_FOLDER_NAME
        output_dir = episode_dir / OUTPUT_DICT_FOLDER_NAME
        obs_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        for i, (obs_dict, output_dict) in enumerate(zip(self.observations_list, self.outputs_list)):
            # Filter out keys containing "visual_feature" from obs_dict
            filtered_obs_dict = {k: v for k, v in obs_dict.items() if "visual_feature" not in k}
            # Use savez_compressed to save dictionaries properly - allows dict-style access when loading
            np.savez(obs_dir / f"{i:06}.npz", **filtered_obs_dict)
            np.savez(output_dir / f"{i:06}.npz", **output_dict)

    def write(self, success: Optional[bool] = None):
        """Write out the data to a file."""

        now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

        # Create the episode directory
        episode_dir = self.task_dir / now
        episode_dir.mkdir(exist_ok=True)
        print("Processing data from the dictionary...")
        self.write_recorded_dict(episode_dir)
        self.process_data()

        # Write the images
        print("Write end effector camera feed...")
        for i, (rgb, depth) in tqdm(enumerate(zip(self.rgbs, self.depths)), ncols=80):
            if rgb is None or depth is None:
                continue
            self.write_image(rgb, depth, episode_dir, i)

        print("Write head camera feed...")
        for i, (rgb, depth) in tqdm(enumerate(zip(self.head_rgbs, self.head_depths)), ncols=80):
            if rgb is None or depth is None:
                continue
            self.write_image(rgb, depth, episode_dir, i, head=True)

        # Run video processing
        print("Processing end effector camera feed...")
        self.process_rgb_to_video(episode_dir)
        self.process_depth_to_bin(episode_dir)
        print("Processing head camera feed...")
        self.process_rgb_to_video(episode_dir, head=True)
        self.process_depth_to_bin(episode_dir, head=True)

        print("Writing metadata...")

        # Write a file saying this is done
        with open(str(episode_dir / "completed.txt"), "w") as file:
            # Write the string to the file
            file.write("Completed")

        # We only write success if it is explicitly provided
        if success is not None:
            # Write success or failure
            with open(str(episode_dir / "success.txt"), "w") as file:
                # Write the string to the file
                if success:
                    file.write("Success")
                else:
                    file.write("Failure")

        with open(episode_dir / "labels.json", "w") as f:
            json.dump(self.data_dicts, f)

        # Add episode info to metadata
        self.metadata["date"] = now
        self.metadata["num_frames"] = len(self.rgbs)

        # Collect git information if it exists
        self.metadata["git_branch"] = git_tools.get_git_branch()
        self.metadata["git_commit"] = git_tools.get_git_commit()
        self.metadata["git_commit_message"] = git_tools.get_git_commit_message()

        # Write metadata json file
        # Convert metadata to JSON-serializable format (handles OmegaConf, edict, numpy arrays, etc.)
        serializable_metadata = make_json_serializable(self.metadata)
        with open(str(episode_dir / "configs.json"), "w") as fp:
            json.dump(serializable_metadata, fp, indent=2)

        if not self.save_images:
            self.cleanup_image_folders(episode_dir)

        # Reset the recorder
        self.reset()
        print("Done!")


