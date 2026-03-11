# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import pprint as pp
import time

import cv2
import numpy as np

import stretch.app.dex_teleop.dex_teleop_parameters as dt
import stretch.app.dex_teleop.dex_teleop_utils as dt_utils
import stretch.app.dex_teleop.goal_from_teleop as gt
import stretch.app.dex_teleop.webcam_teleop_interface as wt
import stretch.motion.constants as constants
import stretch.motion.simple_ik as si
import stretch.utils.logger as logger
import stretch.utils.loop_stats as lt
from stretch.agent.zmq_client import HomeRobotZmqClient

try:
    from stretch.app.dex_teleop.hand_tracker import HandTracker
except ImportError as e:
    print("Hand tracker not available. Please install its dependencies if you want to use it.")
    print()
    print("\tpython -m pip install .[hand_tracker]")
    print()
from stretch.core import get_parameters
from stretch.motion.kinematics import HelloStretchIdx
from stretch.utils.data_tools.record import FileDataRecorder
from stretch.app.lfd.policy_utils import normalize_gripper, unnormalize_gripper
HOME_POS = np.array([-0.025, -0.35, 0.6])

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

class ZmqRos2Leader:
    """Leader class for DexTeleop using the Zmq_client for ROS2 on Stretch"""

    def __init__(
        self,
        robot: HomeRobotZmqClient,
        verbose: bool = False,
        left_handed: bool = False,
        data_dir: str = "./data",
        task_name: str = "task",
        user_name: str = "default_user",
        env_name: str = "default_env",
        force_record: bool = False,
        debug_aruco: bool = False,
        save_images: bool = False,
        teleop_mode: str = "base_x",
        record_success: bool = False,
        platform: str = "linux",
        use_clutch: bool = False,
        teach_grasping: bool = False,
        teleop_factor: float = 0.5,
        perf_debug: bool = False,
    ):
        self.robot = robot
        self.camera = None

        # TODO: fix these two things
        manipulate_on_ground = False
        slide_lift_range = False
        self.save_images = save_images
        self.teleop_mode = teleop_mode
        self.record_success = record_success
        self.platform = platform
        self.verbose = verbose
        self.use_clutch = use_clutch
        self.teach_grasping = teach_grasping
        self.teleop_factor = teleop_factor
        self.perf_debug = perf_debug

        self.left_handed = left_handed

        self.base_x_origin = None
        self.current_base_x = 0.0

        self.use_gripper_center = True

        lift_middle = dt.get_lift_middle(manipulate_on_ground)
        center_configuration = dt.get_center_configuration(lift_middle)
        starting_configuration = dt.get_starting_configuration(lift_middle)

        if debug_aruco:
            logger.warning(
                "Debugging aruco markers. This displays an OpenCV UI which may make it difficult to enter commands. Do not use this option when doing data collection."
            )
        if left_handed:
            self.webcam_aruco_detector = wt.WebcamArucoDetector(
                tongs_prefix="left",
                visualize_detections=False,
                show_debug_images=debug_aruco,
                platform=platform,
            )
        else:
            self.webcam_aruco_detector = wt.WebcamArucoDetector(
                tongs_prefix="right",
                visualize_detections=False,
                show_debug_images=debug_aruco,
                platform=platform,
            )

        # Get Wrist URDF joint limits
        translation_urdf_file_name = "./stretch_base_translation_ik_with_fixed_wrist.urdf"
        translation_urdf = dt_utils.load_urdf(translation_urdf_file_name)
        wrist_joints = ["joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll"]
        self.wrist_joint_limits = {}
        for joint_name in wrist_joints:
            joint = translation_urdf.joint_map.get(joint_name, None)
            if joint is not None:
                lower = float(joint.limit.lower)
                upper = float(joint.limit.upper)
                self.wrist_joint_limits[joint.name] = (lower, upper)

        self.drop_extreme_wrist_orientation_change = True

        # Initialize the filtered wrist orientation that is used to
        # command the robot. Simple exponential smoothing is used to
        # filter wrist orientation values coming from the interface
        # objects.
        self.filtered_wrist_orientation = np.array([0.0, 0.0, 0.0])

        # Initialize the filtered wrist position that is used to command
        # the robot. Simple exponential smoothing is used to filter wrist
        # position values coming from the interface objects.
        self.filtered_wrist_position_configuration = np.array(
            [
                starting_configuration["joint_mobile_base_rotate_by"],
                starting_configuration["joint_lift"],
                starting_configuration["joint_arm_l0"],
                # starting_configuration["joint_arm_l1"],
                # starting_configuration["joint_arm_l2"],
                # starting_configuration["joint_arm_l3"],
            ]
        )

        self.prev_commanded_wrist_orientation = {
            "joint_wrist_yaw": None,
            "joint_wrist_pitch": None,
            "joint_wrist_roll": None,
        }

        # This is the weight multiplied by the current wrist angle command when performing exponential smoothing.
        # 0.5 with 'max' robot speed was too noisy on the wrist
        self.wrist_orientation_filter = dt.exponential_smoothing_for_orientation

        # This is the weight multiplied by the current wrist position command when performing exponential smoothing.
        # commands before sending them to the robot
        self.wrist_position_filter = dt.exponential_smoothing_for_position

        self.print_robot_status_thread_timing = False
        self.debug_wrist_orientation = False

        self.max_allowed_wrist_yaw_change = dt.max_allowed_wrist_yaw_change
        self.max_allowed_wrist_roll_change = dt.max_allowed_wrist_roll_change

        # Initialize simple IK
        simple_ik = si.SimpleIK()

        # Define the center position for the wrist that corresponds with
        # the teleop origin.
        self.center_wrist_position = simple_ik.fk_rotary_base(center_configuration)

        self.goal_from_markers = gt.GoalFromMarkers(
            dt.teleop_origin,
            self.center_wrist_position,
            slide_lift_range=slide_lift_range,
        )

        # Save metadata to pass to recorder
        self.metadata = {
            "recording_type": "Dex Teleop",
            "user_name": user_name,
            "task_name": task_name,
            "env_name": env_name,
            "left_handed": left_handed,
            "teleop_mode": teleop_mode,
            "backend": "ros2",
        }

        self._force = force_record
        self._recording = False or self._force
        self._need_to_write = False
        self._recorder = FileDataRecorder(
            data_dir, task_name, user_name, env_name, save_images, self.metadata, fps=6
        )
        self.prev_goal_dict = None

    def ask_for_success(self) -> bool:
        """Ask the user if the episode was successful."""
        while True:
            logger.alert("Was the episode successful? (y/n)")
            key = cv2.waitKey(0)
            if key == ord("y") or key == ord("Y"):
                return True
            elif key == ord("n") or key == ord("N"):
                return False

    def get_goal_joint_config(
        self,
        grip_width,
        wrist_position: np.ndarray,
        gripper_orientation: np.ndarray,
        relative: bool = False,
        verbose: bool = False,
        **config,
    ):
        current_joint_positions = self.robot.get_joint_positions()
        full_body_cfg, success, info = self.robot._robot_model.manip_ik(
            (wrist_position, gripper_orientation),
            q0=current_joint_positions,
        )

        if full_body_cfg is None or not success:
            new_goal_configuration = None
        else:
            manip_joint_positions = self.robot._extract_joint_pos(full_body_cfg)
            new_goal_configuration = {
                "joint_fake": full_body_cfg[HelloStretchIdx.BASE_X],
                "joint_lift": full_body_cfg[HelloStretchIdx.LIFT],
                "joint_arm_l0": manip_joint_positions[2],
                "joint_wrist_yaw": full_body_cfg[HelloStretchIdx.WRIST_YAW],
                "joint_wrist_pitch": full_body_cfg[HelloStretchIdx.WRIST_PITCH],
                "joint_wrist_roll": full_body_cfg[HelloStretchIdx.WRIST_ROLL],
            }

        if not success:
            print("!!! BAD IK SOLUTION !!!")
            new_goal_configuration = None
        if verbose:
            pp.pp(new_goal_configuration)

        if new_goal_configuration is None:
            print(
                f"WARNING: IK failed to find a valid new_goal_configuration so skipping this iteration by continuing the loop. Input to IK: wrist_position = {wrist_position}, Output from IK: new_goal_configuration = {new_goal_configuration}"
            )
        else:
            # Use the same aggregate manip joint representation as arm_to_ee_pose().
            new_wrist_position_configuration = np.array(
                [
                    new_goal_configuration["joint_fake"],
                    new_goal_configuration["joint_lift"],
                    new_goal_configuration["joint_arm_l0"],
                ]
            )
            
            # TEST: Print IK output for debugging
            if verbose:
                print(f"[IK TEST] Direct IK output:")
                print(f"  base_x: {new_goal_configuration['joint_fake']:.4f}")
                print(f"  lift: {new_goal_configuration['joint_lift']:.4f}")
                print(f"  arm: {new_goal_configuration['joint_arm_l0']:.4f}")
                print(f"  wrist_yaw: {new_goal_configuration['joint_wrist_yaw']:.4f}")
                print(f"  wrist_pitch: {new_goal_configuration['joint_wrist_pitch']:.4f}")
                print(f"  wrist_roll: {new_goal_configuration['joint_wrist_roll']:.4f}")

            # Use exponential smoothing to filter the wrist
            # position configuration used to command the
            # robot.
            self.filtered_wrist_position_configuration = (
                (1.0 - self.wrist_position_filter) * self.filtered_wrist_position_configuration
            ) + (self.wrist_position_filter * new_wrist_position_configuration)

            if self.teleop_mode == "base_x":
                new_goal_configuration["base_x_joint"] = (
                    self.filtered_wrist_position_configuration[0] * self.teleop_factor
                )
                new_goal_configuration["joint_mobile_base_rotate_by"] = 0.0
                new_goal_configuration["joint_mobile_base_translate_by"] = 0.0
            elif self.teleop_mode == "rotary_base":
                new_goal_configuration[
                    "joint_mobile_base_rotate_by"
                ] = self.filtered_wrist_position_configuration[0]
                new_goal_configuration["base_x_joint"] = 0.0
                new_goal_configuration["joint_mobile_base_translate_by"] = 0.0
            else:
                new_goal_configuration["joint_mobile_base_rotate_by"] = 0.0
                new_goal_configuration["base_x_joint"] = 0.0
                new_goal_configuration["joint_mobile_base_translate_by"] = 0.0

            new_goal_configuration["joint_lift"] = self.filtered_wrist_position_configuration[1]
            new_goal_configuration["joint_arm_l0"] = self.filtered_wrist_position_configuration[2]
            # new_goal_configuration["joint_arm_l1"] = self.filtered_wrist_position_configuration[3]
            # new_goal_configuration["joint_arm_l2"] = self.filtered_wrist_position_configuration[4]
            # new_goal_configuration["joint_arm_l3"] = self.filtered_wrist_position_configuration[5]

            #################################

            #################################
            # INPUT: grip_width between 0.0 and 1.0, here unnormalized to [Gripper_MIN, Gripper_MAX]
            if (grip_width is not None) and (grip_width > -1000.0):
                # Use width to interpolate between open and closed
                new_goal_configuration["stretch_gripper"] = (
                    self.robot._robot_model.GRIPPER_CLOSED
                ) + grip_width * (
                    abs(self.robot._robot_model.GRIPPER_OPEN)
                    + abs(self.robot._robot_model.GRIPPER_CLOSED)
                )

            ##################################################
            # INPUT: x_axis, y_axis, z_axis

            wrist_yaw = new_goal_configuration["joint_wrist_yaw"]
            wrist_pitch = new_goal_configuration["joint_wrist_pitch"]
            wrist_roll = new_goal_configuration["joint_wrist_roll"]

            if self.debug_wrist_orientation:
                print("___________")
                print(
                    "wrist_yaw, wrist_pitch, wrist_roll = {:.2f}, {:.2f}, {:.2f} deg".format(
                        (180.0 * (wrist_yaw / np.pi)),
                        (180.0 * (wrist_pitch / np.pi)),
                        (180.0 * (wrist_roll / np.pi)),
                    )
                )

            limits_violated = False
            lower_limit, upper_limit = self.wrist_joint_limits["joint_wrist_yaw"]
            if (wrist_yaw < lower_limit) or (wrist_yaw > upper_limit):
                limits_violated = True
            lower_limit, upper_limit = self.wrist_joint_limits["joint_wrist_pitch"]
            if (wrist_pitch < lower_limit) or (wrist_pitch > upper_limit):
                limits_violated = True
            lower_limit, upper_limit = self.wrist_joint_limits["joint_wrist_roll"]
            if (wrist_roll < lower_limit) or (wrist_roll > upper_limit):
                limits_violated = True

            ################################################################
            # DROP GRIPPER ORIENTATION GOALS WITH LARGE JOINT ANGLE CHANGES
            #
            # Dropping goals that result in extreme changes in joint
            # angles over a single time step avoids the nearly 360
            # degree rotation in an opposite direction of motion that
            # can occur when a goal jumps across a joint limit for a
            # joint with a large range of motion like the roll joint.
            #
            # This also reduces the potential for unexpected wrist
            # motions near gimbal lock when the yaw and roll axes are
            # aligned (i.e., the gripper is pointed down to the
            # ground). Goals representing slow motions that traverse
            # near this gimbal lock region can still result in the
            # gripper approximately going upside down in a manner
            # similar to a pendulum, but this results in large yaw
            # joint motions and is prevented at high speeds due to
            # joint angles that differ significantly between time
            # steps. Inverting this motion must also be performed at
            # low speeds or the gripper will become stuck and need to
            # traverse a trajectory around the gimbal lock region.
            #
            extreme_difference_violated = False
            if self.drop_extreme_wrist_orientation_change:
                prev_wrist_yaw = self.prev_commanded_wrist_orientation["joint_wrist_yaw"]
                if prev_wrist_yaw is not None:
                    diff = abs(wrist_yaw - prev_wrist_yaw)
                    if diff > self.max_allowed_wrist_yaw_change:
                        print(
                            "extreme wrist_yaw change of {:.2f} deg".format(
                                (180.0 * (diff / np.pi))
                            )
                        )
                        extreme_difference_violated = True
                prev_wrist_roll = self.prev_commanded_wrist_orientation["joint_wrist_roll"]
                if prev_wrist_roll is not None:
                    diff = abs(wrist_roll - prev_wrist_roll)
                    if diff > self.max_allowed_wrist_roll_change:
                        print(
                            "extreme wrist_roll change of {:.2f} deg".format(
                                (180.0 * (diff / np.pi))
                            )
                        )
                        extreme_difference_violated = True
            #
            ################################################################

            if self.debug_wrist_orientation:
                if limits_violated:
                    print("The wrist angle limits were violated.")

            if (not extreme_difference_violated) and (not limits_violated):
                new_wrist_orientation = np.array([wrist_yaw, wrist_pitch, wrist_roll])

                # Use exponential smoothing to filter the wrist
                # orientation configuration used to command the
                # robot.
                self.filtered_wrist_orientation = (
                    (1.0 - self.wrist_orientation_filter) * self.filtered_wrist_orientation
                ) + (self.wrist_orientation_filter * new_wrist_orientation)

                new_goal_configuration["joint_wrist_yaw"] = self.filtered_wrist_orientation[0]
                new_goal_configuration["joint_wrist_pitch"] = self.filtered_wrist_orientation[1]
                new_goal_configuration["joint_wrist_roll"] = self.filtered_wrist_orientation[2]

                self.prev_commanded_wrist_orientation = {
                    "joint_wrist_yaw": self.filtered_wrist_orientation[0],
                    "joint_wrist_pitch": self.filtered_wrist_orientation[1],
                    "joint_wrist_roll": self.filtered_wrist_orientation[2],
                }

            # Teleop mode specific modifications
            # if self.teleop_mode == "base_x":
            #     new_goal_configuration["joint_mobile_base_rotate_by"] = 0.0

            #     # Base_x_origin is reset to base_x coordinate at start of demonstration
            #     if self.base_x_origin is None:
            #         self.base_x_origin = current_state["base_x"]

            #     self.current_base_x = current_state["base_x"] - self.base_x_origin

            #     new_goal_configuration["joint_mobile_base_translate_by"] = (
            #         new_goal_configuration["joint_mobile_base_translation"] - self.current_base_x
            #     )

        return new_goal_configuration

    def run(self, display_received_images, loop_rate: float = 10.0):
        loop_timer = lt.LoopStats("dex_teleop_leader")

        if self.use_clutch:
            hand_tracker = HandTracker(left_clutch=(not self.left_handed))

        print("=== Starting Dex Teleop Leader ===")
        print("Press spacebar to start/stop recording.")
        if self.teach_grasping:
            print("Press 1, 2, or 3 to teach PREGRASP, GRASP, or POSTGRASP.")
        print("Press 0-9 to record waypoints.")
        if self.record_success:
            print("Press y/n to record success/failure of episode after each episode.")
        if self.use_clutch:
            print("Clutch mode enabled. Place an empty hand over the webcam to clutch.")
        print("Press ESC to exit.")

        # loop stuff for clutch
        clutched = False
        clutch_debounce_threshold = 3
        change_clutch_count = 0
        check_hand_frame_skip = 3
        i = 0
        max_i = 100  # arbitrary number of iterations
        last_recorded_servo_seq = None
        
        if self.perf_debug:
            perf_last_print = time.perf_counter()
            perf_loop_count = 0
            perf_new_obs_count = 0
            perf_record_count = 0  # Track number of recordings
            perf_last_obs_id = None
            perf_last_servo_seq = None

        # last_robot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # offset_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        last_robot_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        offset_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        self.robot.reset_manipulation_base_pose()
        print('reset robot manip base pose!')

        try:
            while True:
                loop_start_time = time.time()
                waypoint_key = None

                loop_timer.mark_start()

                # Get observation
                if self.perf_debug:
                    perf_loop_count += 1
                    observation = self.robot.get_servo_observation()
                    servo_seq, _, _ = self.robot.get_servo_stats()
                    if perf_last_obs_id is None or servo_seq != perf_last_obs_id:
                        perf_new_obs_count += 1
                        perf_last_obs_id = servo_seq
                else:
                    observation = self.robot.get_servo_observation()
                    servo_seq, _, _ = self.robot.get_servo_stats()

                # Process images
                gripper_color_image = cv2.cvtColor(observation.ee_rgb, cv2.COLOR_RGB2BGR) # BGR
                gripper_depth_image = observation.ee_depth.astype(np.float32)

                # print('gripper cam shape', gripper_color_image.shape)
                # gripper_cam_pose = observation.ee_camera_pose
                # gripper_cam_K = observation.ee_camera_K

                head_color_image = cv2.cvtColor(observation.rgb, cv2.COLOR_RGB2BGR)
                # print('head cam shape', head_color_image.shape)
                # print('depth scaling', observation.depth_scaling)
                head_depth_image = observation.depth.astype(np.float32) 

                if display_received_images:
                    # change depth to be h x w x 3
                    depth_image_x3 = np.stack((gripper_depth_image,) * 3, axis=-1)
                    combined = np.hstack((gripper_color_image / 255, depth_image_x3 / 4))

                    # Head images
                    head_depth_image_x3 = np.stack((head_depth_image,) * 3, axis=-1)
                    head_combined = np.hstack((head_color_image / 255, head_depth_image_x3 / 4))

                    # Get the current height and width
                    (height, width) = combined.shape[:2]
                    (head_height, head_width) = head_combined.shape[:2]

                    # Calculate the aspect ratio
                    aspect_ratio = float(head_width) / float(head_height)

                    # Calculate the new height based on the aspect ratio
                    new_height = int(width / aspect_ratio)

                    head_combined = cv2.resize(
                        head_combined, (width, new_height), interpolation=cv2.INTER_LINEAR
                    )

                    # Combine both images from ee and head
                    combined = np.vstack((combined, head_combined))
                    downsample_ratio = 0.5
                    current_width = combined.shape[1]
                    current_height = combined.shape[0]
                    combined_downsampled = cv2.resize(combined, (int(current_width * downsample_ratio), int(current_height * downsample_ratio)), interpolation=cv2.INTER_NEAREST)    
                    cv2.imshow("Observed RGB/Depth Image", combined_downsampled)

                if self.perf_debug:
                    now = time.perf_counter()
                    dt = now - perf_last_print
                    if dt >= 2.0:
                        loop_rate = perf_loop_count / dt
                        new_obs_rate = perf_new_obs_count / dt
                        record_rate = perf_record_count / dt  # Compute recording frequency
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
                                f"[PERF] loop={loop_rate:.2f} Hz, new_obs={new_obs_rate:.2f} Hz, record={record_rate:.2f} Hz, servo_age={servo_age_str}"
                            )
                        else:
                            print(
                                f"[PERF] loop={loop_rate:.2f} Hz, new_obs={new_obs_rate:.2f} Hz, servo={servo_rate:.2f} Hz, record={record_rate:.2f} Hz, servo_age={servo_age_str}"
                            )
                        perf_last_print = now
                        perf_loop_count = 0
                        perf_new_obs_count = 0
                        perf_record_count = 0  # Reset recording counter

                # Wait for spacebar to be pressed and start/stop recording
                # Spacebar is 32
                # Escape is 27
                key = cv2.waitKey(1)
                if key == 32:
                    self._recording = not self._recording
                    self.prev_goal_dict = None
                    print('reset robot manip base pose!')
                    if self._recording:
                        # Reset base_x_origin
                        self.base_x_origin = None
                        print("[LEADER] Recording started.")
                    else:
                        print("[LEADER] Recording stopped.")
                        self._need_to_write = True
                        if self._force:
                            # Try to terminate
                            print("[LEADER] Force recording done. Terminating.")
                            return None
                elif key == 27:
                    if self._recording:
                        self._need_to_write = True
                    self._recording = False
                    print("[LEADER] Recording stopped. Terminating.")
                    break
                else:
                    for i in range(10):
                        if key == ord(str(i)):
                            if self.teach_grasping and i >= 1 and i <= 3:
                                if i == 1:
                                    print(f"[LEADER] Key {i} pressed. Teaching PREGRASP.")
                                elif i == 2:
                                    print(f"[LEADER] Key {i} pressed. Teaching GRASP.")
                                elif i == 3:
                                    print(f"[LEADER] Key {i} pressed. Teaching POSTGRASP.")
                            else:
                                print(f"[LEADER] Key {i} pressed. Recording waypoint {i}.")
                            waypoint_key = i
                            break

                # Raw input from teleop
                markers, color_image = self.webcam_aruco_detector.process_next_frame()
                if color_image is None:
                    "Waiting for webcam images!"
                    time.sleep(0.05)
                    continue

                # Set up commands to be sent to the robot
                goal_dict = self.goal_from_markers.get_goal_dict(markers)

                if self.use_clutch:
                    # check if n-th frame - if so, check clutch
                    if i % check_hand_frame_skip == 0:
                        hand_prediction = hand_tracker.run_detection(color_image)
                        check_clutched = hand_tracker.check_clutched(hand_prediction)

                        # debounce
                        if check_clutched != clutched:
                            change_clutch_count += 1
                        else:
                            change_clutch_count = 0

                        if change_clutch_count >= clutch_debounce_threshold:
                            clutched = not clutched
                            change_clutch_count = 0

                    i += 1
                    i = i % max_i

                if goal_dict is not None:
                    # Convert goal dict into a quaternion
                    goal_dict = dt_utils.process_goal_dict(
                        goal_dict, self.prev_goal_dict, self.use_gripper_center
                    )
                else:
                    # Goal dict that is not worth processing
                    goal_dict = {"valid": False}

                if goal_dict["valid"]:
                    # get robot configuration
                    goal_configuration = self.get_goal_joint_config(**goal_dict)
                    if goal_configuration is None:
                        continue

                    # Format to standard action space
                    goal_configuration = dt_utils.format_actions(goal_configuration)

                    if self._recording:
                        print("[LEADER] goal_dict =")
                        pp.pprint(goal_configuration)

                    robot_pose = np.array(
                        [
                            goal_configuration["base_x_joint"],
                            goal_configuration["joint_lift"],
                            goal_configuration["joint_arm_l0"],
                            # goal_configuration["joint_arm_l1"],
                            # goal_configuration["joint_arm_l2"],
                            # goal_configuration["joint_arm_l3"],
                            goal_configuration["joint_wrist_yaw"],
                            goal_configuration["joint_wrist_pitch"],
                            goal_configuration["joint_wrist_roll"],
                        ]
                    )

                    if not clutched:
                        # Prep joint states as dict
                        if self._recording and self.prev_goal_dict is not None:
                            current_servo_seq = servo_seq
                            if current_servo_seq != last_recorded_servo_seq:
                                joint_states = {
                                    k: observation.joint[v] for k, v in HelloStretchIdx.name_to_idx.items()
                                }
                                gripper_state = joint_states['gripper']
                                gripper_state_normalized = normalize_gripper(gripper_state) # to [0, 1]
                                observation_dict = {
                                    "joint_states": joint_states,
                                    "ee_pose": observation.ee_pose.tolist(),
                                    "gripper": gripper_state_normalized.tolist()
                                }
                                action_dict = {
                                    "joint_goal_configuration": goal_configuration,
                                    "ee_goal_pose": goal_dict['absolute_gripper_pose'].tolist(),
                                    "gripper_goal": goal_dict["grip_width"].tolist(), # [0,1]
                                }
                                # Only record if the observation object is different from the last one we saved
                                self._recorder.add(
                                    ee_rgb=observation.ee_rgb, # RGB
                                    ee_depth=observation.ee_depth, # meters
                                    ee_cam_pose=observation.ee_camera_pose,
                                    ee_cam_K=observation.ee_camera_K,
                                    xyz=goal_dict["relative_gripper_position"],
                                    quaternion=goal_dict["relative_gripper_orientation"],
                                    ee_goal_pose=goal_dict['absolute_gripper_pose'],
                                    gripper=goal_dict["grip_width"],
                                    ee_pose=observation.ee_pose,
                                    observations=observation_dict,  # put actual states: ee_pose, normalized gripper
                                    actions=action_dict, # goal joint configuration, goal pose, goal gripper
                                    head_rgb=observation.rgb, # RGB
                                    head_depth=head_depth_image,
                                    head_cam_pose=observation.camera_pose,
                                    head_cam_K=observation.camera_K,
                                )

                                last_robot_pose = robot_pose
                                # add clutch offset
                                robot_pose += offset_pose

                                self.robot.arm_to(
                                    robot_pose,
                                    gripper=goal_configuration["stretch_gripper"], # [Gripper_MIN, Gripper_MAX]
                                    head=constants.look_at_ee,
                                    blocking=False,  # We set this flag to False to make sure it doesn't block
                                    reliable=False,  # We set this flag to False so we dont wait for receipt
                                )
                                if self.perf_debug:
                                    perf_record_count += 1  # Increment recording counter
                                    print(f'[PERF] recorded servo_seq: {current_servo_seq}')
                                last_recorded_servo_seq = current_servo_seq

                            # Record waypoint
                            if waypoint_key is not None:
                                print("[LEADER] Recording waypoint.")
                                ok = self._recorder.add_waypoint(
                                    waypoint_key,
                                    robot_pose,
                                    goal_configuration["stretch_gripper"],
                                )
                                if not ok:
                                    logger.warning(
                                        f"[LEADER] WARNING: overwriting previous waypoint {waypoint_key}."
                                    )


                self.prev_goal_dict = goal_dict

                if self.verbose:
                    loop_timer.mark_end()
                    loop_timer.pretty_print()

                if self._need_to_write:
                    if self.record_success:
                        success = self.ask_for_success()
                        print("[LEADER] Writing data to disk with success = ", success)
                        # self.robot.reset_manipulation_base_pose()
                        self._recorder.write(success=success)
                    else:
                        print("[LEADER] Writing data to disk.")
                        self._recorder.write()
                        # self.robot.reset_manipulation_base_pose()
                    self._need_to_write = False
                
                elapsed_time = time.time() -  loop_start_time
                sleep_time = 1 / loop_rate - elapsed_time
                if sleep_time < 0:
                    print(f'WARNING!!!!!! ===================> sleep time is negative: {sleep_time:.3f}s, skipping sleep')
                else: 
                    print(f'sleeping for {sleep_time:.3f}s to maintain {loop_rate}Hz loop rate')
                precise_sleep(max(sleep_time, 0)) # sleep for 0.1s to maintain 5Hz loop rate


        finally:
            print("Exiting...")
            # Go to initial pose
            open = input("Open the gripper: Y/N?")
            if open == "Y" or open == "y":
                self.robot.arm_to_ee_pose(
                    pos = HOME_POS,
                    quat = None, 
                    gripper = 1.0, 
                    world_frame = False,
                    reliable = True,
                    blocking = True,
                )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--robot_ip", type=str, default="")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-u", "--user-name", type=str, default="default_user")
    parser.add_argument("-t", "--task-name", type=str, default="default_task")
    parser.add_argument("-e", "--env-name", type=str, default="default_env")
    parser.add_argument("-f", "--force", action="store_true", help="Force data recording.")
    parser.add_argument("-d", "--data-dir", type=str, default="./data")
    parser.add_argument(
        "-s",
        "--skip-images",
        action="store_true",
        help="Do not save raw images in addition to videos",
    )
    parser.add_argument("-P", "--send_port", type=int, default=4402, help="Port to send goals to.")
    parser.add_argument(
        "--teleop-mode",
        "--teleop_mode",
        type=str,
        default="base_x",
        choices=["stationary_base", "rotary_base", "base_x"],
    )
    parser.add_argument(
        "--skip-success", action="store_true", help="Do not record success of episode."
    )
    parser.add_argument("--show-aruco", action="store_true", help="Show aruco debug information.")
    parser.add_argument("--platform", type=str, default="linux", choices=["linux", "not_linux"])
    parser.add_argument("-c", "--clutch", action="store_true")
    parser.add_argument("--teach-grasping", action="store_true")
    parser.add_argument("--loop_rate", type=int, default=10, help="Loop rate.")
    parser.add_argument("--perf_debug", action="store_true", help="Print loop/servo rates.")
    args = parser.parse_args()

    # Parameters
    MANIP_MODE_CONTROLLED_JOINTS = dt_utils.get_teleop_controlled_joints(args.teleop_mode)
    parameters = get_parameters("default_planner.yaml")

    # Zmq client
    robot = HomeRobotZmqClient(
        robot_ip=args.robot_ip,
        send_port=args.send_port,
        parameters=parameters,
        manip_mode_controlled_joints=MANIP_MODE_CONTROLLED_JOINTS,
        enable_rerun_server=False,
    )
    robot.switch_to_manipulation_mode()
    robot.move_to_manip_posture()

    leader = ZmqRos2Leader(
        robot=robot,
        verbose=args.verbose,
        data_dir=args.data_dir,
        user_name=args.user_name,
        task_name=args.task_name,
        env_name=args.env_name,
        force_record=args.force,
        save_images=(not args.skip_images),
        teleop_mode=args.teleop_mode,
        record_success=(not args.skip_success),
        platform=args.platform,
        use_clutch=args.clutch,
        teach_grasping=args.teach_grasping,
        perf_debug=args.perf_debug,
    )

    try:
        leader.run(display_received_images=True, loop_rate=args.loop_rate)
    except KeyboardInterrupt:
        pass

    robot.stop()
