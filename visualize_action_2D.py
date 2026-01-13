import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import json
import viser  
import scipy.spatial.transform as tra
from stretch.motion.kinematics import HelloStretchKinematics, HelloStretchIdx

HEAD_CAM_INTRINSICS = np.array([[303.6978759765625, 0.0, 125.8017578125], [0.0, 303.5592956542969, 155.52696228027344], [0.0, 0.0, 1.0]])
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

def visualize_projected_pixels(preds_pixels: list, ee_cam_pixels: list, image: np.ndarray) -> np.ndarray:
    """
    Visualize projected pixels on the image using OpenCV.
    Args:
        preds_pixels: (N, 2) array of projected pixels for predictions, from joints action
        targets_pixels: (N, 2) array of projected pixels for targets, from joints gt
        ee_cam_pixels: (N, 2) array of projected pixels for end effector, from observations
        image: (H, W, 3) array of image (RGB or BGR)
    Returns:
        np.ndarray: The resulting image with projected points drawn on it.
    """
    # Make a copy so we don't overwrite the original
    assert len(preds_pixels)  == len(ee_cam_pixels), "Number of predictions, targets, and end effector pixels should match"
    img = image.copy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img)

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(preds_pixels)))

    for i, (u,v) in enumerate(preds_pixels):
        ax.plot(u, v, 'x', markersize=15, label='predictions', color=colors[i])
        ax.plot(ee_cam_pixels[i][0], ee_cam_pixels[i][1], '*', markersize=15, label='ee_projection', color='blue')
    ax.axis('off')
    return fig


def project_pts_in_cam(points: np.ndarray, cam_K: np.ndarray, T_obj_cam: np.ndarray) -> np.ndarray:
    """
    Project 3D points in object frame to 2D pixel coordinates in camera frame
    
    Args:
        points: (N, 3) array of 3D points in object frame
        T_cam_obj: (4, 4) transformation matrix from camera to object frame
        cam_K: (3, 3) camera intrinsic matrix
    
    Returns:
        pixel: (N, 2) array of 2D pixel coordinates
        points_cam: (N, 3) array of 3D points in camera frame
    """
    # Convert points to homogeneous coordinates
    assert points.dtype == cam_K.dtype == T_obj_cam.dtype, "Data types of points: {points.dtype}, cam_K: {cam_K.dtype}, and T_cam_obj: {T_cam_obj.dtype} must match"
    if points.ndim == 1:
        points = points.reshape(1, -1)
    assert points.shape[1] == 3, f"Points must be (N, 3), but instead {points.shape}"
    T_cam_obj = np.linalg.inv(T_obj_cam)

    points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
    points_cam_homo = (T_cam_obj @ points_homo.T).T
    points_cam = points_cam_homo[:, :3] / points_cam_homo[:, 3:4]
    points_2d_homo = (cam_K @ points_cam.T).T
    pixel = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    return pixel, points_cam


### Kinematics
class FKVisualizer:
    def __init__(self, urdf_path: str = ''):
        self.robot_kinematics = HelloStretchKinematics(
            urdf_path=urdf_path,
            ik_type='pinocchio',
            manip_mode_controlled_joints=None,
        )
    
    def visualize(self, joint_states, T_base_cam: np.ndarray) -> list[np.ndarray]:
        projected_pixel_list = []
        pts_ee_list = []
        fk_pos_list = []
        num_horizon = joint_states.shape[0]

        # breakpoint()
        for i in range(num_horizon):
            fk_pos, fk_quat = self.robot_kinematics.manip_fk(joint_states[i], 'gripper_camera_color_optical_frame') #! TEMP, change after get correct ee
            projected_pixel, pts_cam = project_pts_in_cam(fk_pos.reshape(1, 3).astype(np.float32), HEAD_CAM_INTRINSICS.astype(np.float32), T_base_cam.astype(np.float32)) # from base to cam
            projected_pixel_list.append(projected_pixel[0])
            pts_ee_list.append(pts_cam)
            fk_pos_list.append(fk_pos)
        return projected_pixel_list, pts_ee_list, fk_pos_list


    @staticmethod
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


    @staticmethod
    def observations_to_joint_state(observations: dict) -> np.ndarray:
        """Convert observations dictionary to joint state array for HelloStretchKinematics."""
        q = np.zeros(11)  # 11 DOF: base_x, base_y, base_theta, lift, arm, gripper, wrist_roll, wrist_pitch, wrist_yaw, head_pan, head_tilt
        
        # Map observations to joint state indices
        q[HelloStretchIdx.BASE_X] = observations.get('base_x', 0.0)
        q[HelloStretchIdx.BASE_Y] = observations.get('base_y', 0.0)
        q[HelloStretchIdx.BASE_THETA] = observations.get('base_theta', 0.0)
        q[HelloStretchIdx.LIFT] = observations.get('lift', 0.0)
        q[HelloStretchIdx.ARM] = observations.get('arm', 0.0)
        q[HelloStretchIdx.GRIPPER] = observations.get('gripper', observations.get('gripper_finger_right', 0.0))
        q[HelloStretchIdx.WRIST_ROLL] = observations.get('wrist_roll', 0.0)
        q[HelloStretchIdx.WRIST_PITCH] = observations.get('wrist_pitch', 0.0)
        q[HelloStretchIdx.WRIST_YAW] = observations.get('wrist_yaw', 0.0)
        q[HelloStretchIdx.HEAD_PAN] = observations.get('head_pan', 0.0)
        q[HelloStretchIdx.HEAD_TILT] = observations.get('head_tilt', 0.0)

        return q
    
    def actions_to_joint_state(actions: dict) -> np.ndarray:
        """Convert observations dictionary to joint state array for HelloStretchKinematics."""
        q = np.zeros(11)  # 11 DOF: base_x, base_y, base_theta, lift, arm, gripper, wrist_roll, wrist_pitch, wrist_yaw, head_pan, head_tilt
        
        # Map observations to joint state indices
        q[HelloStretchIdx.BASE_X] = actions.get('base_x_joint', 0.0)
        q[HelloStretchIdx.BASE_Y] = actions.get('base_y_joint', 0.0)
        q[HelloStretchIdx.BASE_THETA] = actions.get('base_theta_joint', 0.0)
        q[HelloStretchIdx.LIFT] = actions.get('joint_lift', 0.0)
        q[HelloStretchIdx.ARM] = actions.get('joint_arm_l0', 0.0)
        q[HelloStretchIdx.GRIPPER] = actions.get('stretch_gripper', 0.0)
        q[HelloStretchIdx.WRIST_ROLL] = actions.get('joint_wrist_roll', 0.0)
        q[HelloStretchIdx.WRIST_PITCH] = actions.get('joint_wrist_pitch', 0.0)
        q[HelloStretchIdx.WRIST_YAW] = actions.get('joint_wrist_yaw', 0.0)
        q[HelloStretchIdx.HEAD_PAN] = actions.get('head_pan', 0.0)
        q[HelloStretchIdx.HEAD_TILT] = actions.get('head_tilt', 0.0)

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
    
    DATA_PATH = '/home/wiss/zanr/code/stretch/data/pickup_bottle_v2/default_user/default_env/'
    folder_list = os.listdir(DATA_PATH)
    # randomly select a folder from the list
    selected_folder = os.path.join(DATA_PATH, np.random.choice(folder_list))
    labels_path = os.path.join(selected_folder, 'labels.json')
    
    with open(labels_path, 'r') as f:
        recordings_dict = json.load(f)
    assert str(frame_idx) in recordings_dict, f"Frame {frame_idx} not found in recordings_dict"
    
    # Load camera image
    camera_fn = os.path.join(selected_folder, f'compressed_{camera_name}_images', f'{frame_idx:06d}.png')
    if not os.path.exists(camera_fn):
        raise FileNotFoundError(f"Camera image not found: {camera_fn}")
    else:
        camera_img = cv2.imread(camera_fn, -1)
        camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2RGB)

    joint_states_list = []
    T_base_cam_list = []
    T_base_ee_cam_pos_list = []
    T_base_head_cam = np.array(recordings_dict[str(frame_idx)]['head_cam_pose'])

    for i in range(future_steps):
        # joint_states = recordings_dict[str(frame_idx + i)]['observations']
        # joint_states_list.append(FKVisualizer.observations_to_joint_state(joint_states))
        actions = recordings_dict[str(frame_idx + i)]['actions']
        joint_states_list.append(FKVisualizer.actions_to_joint_state(actions))
        T_base_cam = recordings_dict[str(frame_idx + i)]['ee_cam_pose']
        T_base_cam_list.append(np.array(T_base_cam))
        T_base_ee_cam_pos_list.append(np.array(T_base_cam)[:3, 3])


    projected_pixel_list, pts_ee_list = project_pts_in_cam(np.array(T_base_ee_cam_pos_list), HEAD_CAM_INTRINSICS, T_base_head_cam)
    
    # # set up robot kinematics and compute FK
    fk_visualizer = FKVisualizer()
    projected_pixel_list_FK, pts_ee_list_FK, fk_pos_list = fk_visualizer.visualize(np.array(joint_states_list), T_base_head_cam)
    result_dict = {
        'q': np.array(joint_states_list),
        'fk_pos': np.array(fk_pos_list),
    }
    np.save(os.path.join(selected_folder, f'ee_projection_frame_{frame_idx:06d}_{camera_name}.npy'), result_dict)
    fig = visualize_projected_pixels(projected_pixel_list_FK, projected_pixel_list, camera_img)
    fig.savefig(os.path.join(selected_folder, f'ee_projection_frame_{frame_idx:06d}_{camera_name}.png'))
    print(f"Visualization saved to: {os.path.join(selected_folder, f'ee_projection_frame_{frame_idx:06d}_{camera_name}.png')}")
    fig.show()
    breakpoint()
    plt.close()
