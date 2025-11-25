import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import json
import viser  
import scipy.spatial.transform as tra
from stretch.motion.kinematics import HelloStretchKinematics, HelloStretchIdx

# Utility to add a coordinate frame (as line sets for axes) using viser 'add_polyline' with two-point lines
def add_frame(server, name, T):
    position = T[:3, 3]
    # scipy Rotation.as_quat does not support scalar_first; returns [x, y, z, w]
    xyzw = tra.Rotation.from_matrix(T[:3, :3]).as_quat()
    # Reorder to w, x, y, z (scalar first) for libraries that expect that format
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    server.scene.add_frame(
        name,
        wxyz=wxyz,
        position=position,
    )

def project_pts_in_cam(points: np.ndarray, cam_K: np.ndarray, T_cam_obj: np.ndarray) -> np.ndarray:
    """
    Project 3D points in object frame to 2D pixel coordinates in camera frame.
    
    Args:
        points: (N, 3) array of 3D points in object frame
        T_cam_obj: (4, 4) transformation matrix from object to camera frame
        cam_K: (3, 3) camera intrinsic matrix
    
    Returns:
        (N, 2) array of 2D pixel coordinates
    """
    # Convert points to homogeneous coordinates
    if points.ndim == 1:
        points = points.reshape(1, -1)
    points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Transform points from object frame to camera frame
    points_cam_homo = (T_cam_obj @ points_homo.T).T
    points_cam = points_cam_homo[:, :3] / points_cam_homo[:, 3:4]

    
    # Project to 2D using camera intrinsics
    points_2d_homo = (cam_K @ points_cam.T).T
    pixel = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    
    return pixel, points_cam

def project_end_effector_in_cam(recordings_dict: dict, frame_idx: int, future_steps: int = 0, camera_name: str = 'head') -> list[np.ndarray]:
    """
    Project the end effector position in the camera frame.
    
    Args:
        frame_data: Dictionary containing pose and camera data for a frame
        camera_name: 'gripper' or 'head'
        future_steps: number of future steps to project
    Returns:
        (2,) array of projected pixel coordinates
    """
    # Convert lists to numpy arrays
    # In the data format, ee_pos is actually the full 4x4 pose matrix
    first_frame_data = recordings_dict[str(frame_idx)]
    if camera_name == 'gripper':
        T_base_cam = np.array(first_frame_data['ee_cam_pose'])
        cam_K = np.array(first_frame_data['ee_cam_K'])
    elif camera_name == 'head':
        T_base_cam = np.array(first_frame_data['head_cam_pose'])
        cam_K = np.array(first_frame_data['head_cam_K'])
    else:
        raise ValueError(f"Invalid camera frame name: {camera_name}. Must be 'gripper' or 'head'")
    
    if future_steps > 0:
        projected_pixel_list = []
        pts_ee_list = []
        for i in range(future_steps):
            T_base_ee = np.array(recordings_dict[str(frame_idx + i)]['ee_pos'])     
            T_cam_base = np.linalg.inv(T_base_cam)
            projected_pixel, pts_cam = project_pts_in_cam(T_base_ee[:3, 3].reshape(1, 3), cam_K, T_cam_base) # from base to cam
            projected_pixel_list.append(projected_pixel[0])
            pts_ee_list.append(T_base_ee[:3, 3])
        return projected_pixel_list, pts_ee_list

### Kinematics
def visualize_joint_states(recordings_dict: dict, robot_kinematics: HelloStretchKinematics, frame_idx: int, future_steps: int = 0, camera_name: str = 'head') -> list[np.ndarray]:
    projected_pixel_list = []
    pts_ee_list = []
    first_frame_data = recordings_dict[str(frame_idx)]
    if camera_name == 'gripper':
        T_base_cam = np.array(first_frame_data['ee_cam_pose'])
        cam_K = np.array(first_frame_data['ee_cam_K'])
    elif camera_name == 'head':
        T_base_cam = np.array(first_frame_data['head_cam_pose'])
        cam_K = np.array(first_frame_data['head_cam_K'])
    else:
        raise ValueError(f"Invalid camera frame name: {camera_name}. Must be 'gripper' or 'head'")
    T_cam_base = np.linalg.inv(T_base_cam)

    for i in range(future_steps):
        frame_data = recordings_dict[str(frame_idx + i)]
        
        # Extract recorded EE pose from matrix
        ee_pose_matrix = np.array(frame_data['ee_pos'])
        recorded_pos, recorded_quat = extract_pose_from_matrix(ee_pose_matrix)
        
        # Build joint state from observations
        observations = frame_data['observations']
        joint_state = observations_to_joint_state(observations)
        
        # Compute FK
        fk_pos, fk_quat = robot_kinematics.manip_fk(joint_state, 'gripper_camera_color_optical_frame')
    
        projected_pixel, pts_cam = project_pts_in_cam(fk_pos.reshape(1, 3), cam_K, T_cam_base) # from base to cam
        projected_pixel_list.append(projected_pixel[0])
        pts_ee_list.append(pts_cam)
    return projected_pixel_list, pts_ee_list


def extract_pose_from_matrix(T: np.ndarray):
    """Extract position and quaternion (w, x, y, z) from 4x4 transformation matrix."""
    pos = T[:3, 3]
    # Use scipy.spatial.transform.Rotation to extract quaternion as (x, y, z, w)
    from scipy.spatial.transform import Rotation as R
    rot = R.from_matrix(T[:3, :3])
    quat_xyzw = rot.as_quat()  # (x, y, z, w)
    # Reorder to (w, x, y, z) to match previous convention
    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return pos, quat


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_idx', type=int, default=200)
    parser.add_argument('--future_steps', type=int, default=8)
    parser.add_argument('--camera_name', type=str, default='head')
    args = parser.parse_args()
    frame_idx = args.frame_idx
    future_steps = args.future_steps
    camera_name = args.camera_name
    
    DATA_PATH = '/home/wiss/zanr/code/stretch/data/pickup_bottle_v2/default_user/default_env/2025-11-17--20-00-01'
    labels_path = os.path.join(DATA_PATH, 'labels.json')
    
    with open(labels_path, 'r') as f:
        recordings_dict = json.load(f)
    assert str(frame_idx) in recordings_dict, f"Frame {frame_idx} not found in recordings_dict"
    
    # Load camera image
    camera_fn = os.path.join(DATA_PATH, f'compressed_{camera_name}_images', f'{frame_idx:06d}.png')
    if not os.path.exists(camera_fn):
        raise FileNotFoundError(f"Camera image not found: {camera_fn}")
    else:
        camera_img = cv2.imread(camera_fn, -1)
        camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2RGB)

    # Project the end effector in the camera frame
    projected_pixel_list, pts_ee_list = project_end_effector_in_cam(recordings_dict, frame_idx, future_steps=future_steps, camera_name=camera_name)
    
    # set up robot kinematics and compute FK
    robot_kinematics = HelloStretchKinematics(
            urdf_path='',
            ik_type='pinocchio',
            manip_mode_controlled_joints=None,
        )
    
    projected_pixel_list_FK, pts_ee_list_FK = visualize_joint_states(recordings_dict, robot_kinematics, frame_idx, future_steps=future_steps, camera_name=camera_name)
    
    # Visualize the end effector position on the image
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(projected_pixel_list)))
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(camera_img)
    
    # Draw the projected point
    for i, (u, v) in enumerate(projected_pixel_list):
        ax.plot(u, v, 'o', markersize=15, label='End Effector', color=colors[i])
        ax.plot(projected_pixel_list_FK[i][0], projected_pixel_list_FK[i][1], '*', markersize=15, label='End Effector', color=colors[i])
    
    # Check if point is within image bounds
    h, w = camera_img.shape[:2]
    if 0 <= u < w and 0 <= v < h:
        ax.set_title(f'End Effector Projection (Frame {frame_idx}, {camera_name} camera)\n'
                    f'Pixel: ({u:.1f}, {v:.1f})', fontsize=14)
    else:
        ax.set_title(f'End Effector Projection (Frame {frame_idx}, {camera_name} camera)\n'
                    f'Pixel: ({u:.1f}, {v:.1f}) [OUT OF BOUNDS]', fontsize=14, color='red')
    ax.axis('off')
    
    # Save the visualization
    output_path = os.path.join(DATA_PATH, f'ee_projection_frame_{frame_idx:06d}_{camera_name}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()