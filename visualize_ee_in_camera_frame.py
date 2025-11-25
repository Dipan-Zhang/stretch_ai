import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import json
import viser  
import scipy.spatial.transform as tra
    

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

if __name__ == "__main__":
    # Load the robot model (optional, not used in current implementation)
    # robot = HelloStretchKinematics(urdf_path='./stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf')
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_idx', type=int, default=200)
    parser.add_argument('--future_steps', type=int, default=8)
    parser.add_argument('--camera_name', type=str, default='head')
    args = parser.parse_args()
    
    DATA_PATH = '/home/wiss/zanr/code/stretch/data/pickup_bottle_v2/default_user/default_env/2025-11-17--20-00-01'
    
    frame_idx = args.frame_idx
    future_steps = args.future_steps
    camera_name = args.camera_name
    
    # Load labels.json
    labels_path = os.path.join(DATA_PATH, 'labels.json')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
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

    ############
    server = viser.ViserServer()
    print("Viser visualization running at: http://localhost:8080")

    # World (base) frame at identity
    add_frame(server, "base", np.eye(4))
    add_frame(server, "ee", np.array(recordings_dict[str(frame_idx)]['ee_pos']))
    add_frame(server, "cam", np.array(recordings_dict[str(frame_idx)][f'{camera_name}_cam_pose']))
    
    # Project the end effector in the camera frame
    projected_pixel_list, pts_ee_list = project_end_effector_in_cam(recordings_dict, frame_idx, future_steps=future_steps, camera_name=camera_name)
    
    
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(projected_pixel_list)))

    # visualize pts in viser

    for i, pts_ee in enumerate(pts_ee_list):
        server.scene.add_icosphere(
            name=f"pts_{i}",
            position=pts_ee.reshape(-1),
            radius=0.05,
            color=(colors[i][0], colors[i][1], colors[i][2]),
        )


    breakpoint()
    # Visualize the end effector position on the image
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(camera_img)
    
    # Draw the projected point
    for i, (u, v) in enumerate(projected_pixel_list):
        ax.plot(u, v, 'o', markersize=15, label='End Effector', color=colors[i])
        ax.plot(u, v, 'x', markersize=20, markeredgewidth=3, color=colors[i])
    
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
    
    # Try to show, but don't fail if display is not available
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot (this is OK in headless environments): {e}")
    
    plt.close()