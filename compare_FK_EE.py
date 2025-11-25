"""
Compare the recorded end-effector pose with the computed forward kinematics pose.
This script loads recorded data and validates that FK computation matches the recorded EE poses.
"""

import json
import os
import sys
import numpy as np
import trimesh.transformations as tra
from stretch.motion.kinematics import HelloStretchKinematics, HelloStretchIdx

# Add src to path to import stretch modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))



def extract_pose_from_matrix(T: np.ndarray):
    """Extract position and quaternion (w, x, y, z) from 4x4 transformation matrix."""
    pos = T[:3, 3]
    # Extract quaternion in (w, x, y, z) format
    w, x, y, z = tra.quaternion_from_matrix(T)
    quat = np.array([w, x, y, z])
    return pos, quat


def quaternion_distance(q1: np.ndarray, q2: np.ndarray):
    """
    Compute distance between two quaternions accounting for double cover.
    Returns the minimum angle between the two rotations.
    """
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product (accounting for q and -q representing same rotation)
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    
    # Angle between quaternions
    angle = 2 * np.arccos(dot)
    return angle


def observations_to_joint_state(observations: dict) -> np.ndarray:
    """Convert observations dictionary to joint state array for HelloStretchKinematics."""
    q = np.zeros(11)  # 11 DOF: base_x, base_y, base_theta, lift, arm, gripper, wrist_roll, wrist_pitch, wrist_yaw, head_pan, head_tilt
    
    # Map observations to joint state indices
    q[HelloStretchIdx.BASE_X] = observations.get('base_x', 0.0)
    # q[HelloStretchIdx.BASE_Y] = observations.get('base_y', 0.0)
    # q[HelloStretchIdx.BASE_THETA] = observations.get('base_theta', 0.0)
    q[HelloStretchIdx.LIFT] = observations.get('lift', 0.0)
    q[HelloStretchIdx.ARM] = observations.get('arm', 0.0)
    # q[HelloStretchIdx.GRIPPER] = observations.get('gripper', observations.get('gripper_finger_right', 0.0))
    q[HelloStretchIdx.WRIST_ROLL] = observations.get('wrist_roll', 0.0)
    q[HelloStretchIdx.WRIST_PITCH] = observations.get('wrist_pitch', 0.0)
    q[HelloStretchIdx.WRIST_YAW] = observations.get('wrist_yaw', 0.0)
    q[HelloStretchIdx.HEAD_PAN] = observations.get('head_pan', 0.0)
    q[HelloStretchIdx.HEAD_TILT] = observations.get('head_tilt', 0.0)
    
    return q


def main():
    DATA_PATH = '/home/wiss/zanr/code/stretch/data/pickup_bottle_v2/default_user/default_env/2025-11-17--20-00-01'
    
    # Load labels.json
    labels_path = os.path.join(DATA_PATH, 'labels.json')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    with open(labels_path, 'r') as f:
        recordings_dict = json.load(f)
    
    # Load the URDF file
    urdf_path = ''
    
    manip_mode_controlled_joints = [
        "joint_arm_l0",
        "joint_lift",
        "joint_wrist_yaw",
        "joint_wrist_pitch",
        "joint_wrist_roll",
        "base_x_joint"
    ]
    robot_kinematics = HelloStretchKinematics(
                urdf_path=urdf_path,
                ik_type='pinocchio',
                manip_mode_controlled_joints=None,
            )
    
    # Tolerances for comparison
    pos_tolerance = 1e-2  # 1 cm
    quat_tolerance = 0.01  # ~0.57 degrees in radians
    
    print(f"Comparing recorded EE poses with FK computation...")
    print(f"Position tolerance: {pos_tolerance*1000:.1f} mm")
    print(f"Orientation tolerance: {np.degrees(quat_tolerance):.2f} degrees\n")
    
    num_frames = len(recordings_dict)
    num_passed = 0
    num_failed = 0
    max_pos_error = 0.0
    max_quat_error = 0.0
    
    # Sort frame indices for consistent processing
    frame_indices = sorted([int(k) for k in recordings_dict.keys()])
    
    for frame_idx in frame_indices:
        frame_data = recordings_dict[str(frame_idx)]
        
        # Extract recorded EE pose from matrix
        ee_pose_matrix = np.array(frame_data['ee_pos'])
        recorded_pos, recorded_quat = extract_pose_from_matrix(ee_pose_matrix)
        
        # Build joint state from observations
        observations = frame_data['observations']
        joint_state = observations_to_joint_state(observations)
        
        # Compute FK
        # fk_pos, fk_quat = robot.manip_fk(joint_state, 'link_gripper_finger_left') 
        fk_pos, fk_quat = robot_kinematics.manip_fk(joint_state, 'gripper_camera_color_optical_frame')
        
        # Compare positions
        pos_error = np.linalg.norm(recorded_pos - fk_pos)
        max_pos_error = max(max_pos_error, pos_error)
        
        # Compare quaternions (accounting for double cover)
        quat_error = quaternion_distance(recorded_quat, fk_quat)
        max_quat_error = max(max_quat_error, quat_error)
        # max_quat_error = 0.0
        # quat_error=0.0
        
        # Check if within tolerance
        pos_ok = pos_error < pos_tolerance
        quat_ok = quat_error < quat_tolerance

        
        if pos_ok and quat_ok:
            num_passed += 1
        else:
            num_failed += 1
            print(f"Frame {frame_idx:4d}: FAILED - "
                  f"pos_error: {pos_error*1000:6.2f} mm, "
                  f"quat_error: {np.degrees(quat_error):5.2f} deg")
            if not pos_ok:
                print(f"  Recorded pos: {recorded_pos}")
                print(f"  FK pos:       {fk_pos}")
            if not quat_ok:
                print(f"  Recorded quat: {recorded_quat}")
                print(f"  FK quat:       {fk_quat}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total frames: {num_frames}")
    print(f"  Passed: {num_passed}")
    print(f"  Failed: {num_failed}")
    print(f"  Max position error: {max_pos_error*1000:.2f} mm")
    print(f"  Max quaternion error: {np.degrees(max_quat_error):.2f} degrees")
    print(f"{'='*60}")
    
    if num_failed == 0:
        print("✓ All frames passed validation!")
        return 0
    else:
        print(f"✗ {num_failed} frames failed validation")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)