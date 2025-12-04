import numpy as np
import open3d as o3d
import scipy.spatial.transform as tra
import json
import cv2
import os
import torch
import liblzfse
import stretch.app.lfd.policy_utils as policy_utils

HEAD_CAM_INTRINSICS = np.array([[303.6978759765625, 0.0, 125.8017578125], [0.0, 303.5592956542969, 155.52696228027344], [0.0, 0.0, 1.0]])
EE_CAM_INTRINSICS = np.array([[215.5474090576172, 0.0, 156.63540649414062], [0.0, 215.40567016601562, 122.5594711303711], [0.0, 0.0, 1.0]])

def backproject(depth, intrinsics, instance_mask, NOCS_convention=True):
    """backproject depth image to 3d points
    Args:
        depth: [h, w]
        intrinsics: [3, 3]
        instance_mask: [h, w]
    return: pts: [num_pixel, 3], idxs: [2, num_pixel]
    """
    intrinsics_inv = np.linalg.inv(intrinsics)
    non_zero_mask = depth > 0
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixsel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    if NOCS_convention:
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]
    return pts, idxs

def backproject_with_color(depth, color, intrinsic, mask, NOCS_convention=False):
    "backproject depth to 3d points and get color"
    pts, pts_idx = backproject(depth, intrinsic, mask, NOCS_convention)
    color = (color / 255.0).astype(np.float32)
    colors = color[pts_idx[0], pts_idx[1]]
    return pts, colors

def visualize_points(points, colors=None):
    "take points and return open3d pcd"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_trajectory(policy, relative_motion, visualization_data_path, idx):
    # Load initial ground truth data
    current_state, \
    gripper_color_image_tensor, \
    head_color_image_tensor, \
    gt_ee_pose_list, \
    visualization_data = load_visualization_data(
    visualization_data_path, idx=idx, relative_motion=relative_motion, device='cuda'
    )
    
    policy.reset()
    
    # Run inference with ground truth observations
    current_pose = gt_ee_pose_list[0].copy()
    waypoints = []
    waypoints.append(current_pose)
    
    for i in range(len(gt_ee_pose_list)):
        observations = {
            "observation.state": current_state,
            "observation.images.gripper": gripper_color_image_tensor,
            "observation.images.head": head_color_image_tensor,
        }
        
        with torch.inference_mode():
            raw_action = policy.select_action(observations)
        
        predicted_action = raw_action[0][:8].cpu().numpy()

        if relative_motion:
            T_rel = np.eye(4)
            T_rel[:3, 3] = np.array(predicted_action[:3])  # Translation
            quat_action = np.array(predicted_action[3:7])  # [qx, qy, qz, qw]
            T_rel[:3, :3] = tra.Rotation.from_quat(quat_action).as_matrix()
            current_pose = current_pose @ T_rel
            waypoints.append(current_pose)
        else:
            T_abs = np.eye(4)
            T_abs[:3, 3] = np.array(predicted_action[:3])  # Translation
            quat_action = np.array(predicted_action[3:7])  # [qx, qy, qz, qw]
            T_abs[:3, :3] = tra.Rotation.from_quat(quat_action).as_matrix()
            current_pose = T_abs
            waypoints.append(current_pose)
    
    # visualize waypoints
    waypoints_vis = []   
    for waypoint in waypoints:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        axis.rotate(waypoint[:3, :3])
        axis.translate(waypoint[:3, 3])
        waypoints_vis.append(axis)
    
    waypoints_gt_vis = []
    for i in range(len(gt_ee_pose_list)):
        gt_ee_pose = gt_ee_pose_list[i]
        gt_ee_pose_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        gt_ee_pose_vis.paint_uniform_color([0, 1, 0])
        gt_ee_pose_vis.rotate(gt_ee_pose[:3, :3])
        gt_ee_pose_vis.translate(gt_ee_pose[:3, 3])
        waypoints_gt_vis.append(gt_ee_pose_vis)
    
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    o3d.visualization.draw_geometries([world_frame] + waypoints_vis + waypoints_gt_vis + [visualization_data['head_cam_pointcloud_inbase']])


def load_visualization_data(data_path: str, idx: int, relative_motion: bool, device: str):
    """Load ground truth state, image, and actions from data path for visualization mode."""
    n_frames = len(os.listdir('{}/compressed_head_depths'.format(data_path)))
    DOWNSAMPLE_FACTOR = 4
    chunk_size= 8*DOWNSAMPLE_FACTOR
    if idx + chunk_size >= n_frames:
        idx = n_frames - chunk_size - 1
        print(f"Index {idx} is out of range. Using the last {chunk_size} frames.")

    # Load images
    gripper_color_image = cv2.imread('{}/compressed_gripper_images/{:06d}.png'.format(data_path, idx))
    head_color_image = cv2.imread('{}/compressed_head_images/{:06d}.png'.format(data_path, idx))
    gripper_color_image = cv2.cvtColor(gripper_color_image, cv2.COLOR_BGR2RGB)
    head_color_image = cv2.cvtColor(head_color_image, cv2.COLOR_BGR2RGB)
    gripper_color_image_tensor = policy_utils.prepare_image(gripper_color_image, device)
    head_color_image_tensor = policy_utils.prepare_image(head_color_image, device)

    head_depth_fname= '{}/compressed_np_head_depth_float32.bin'.format(data_path)
    with open(head_depth_fname, 'rb') as f:
        size = head_color_image.shape[:2]
        head_depths = liblzfse.decompress(f.read())
        head_depths = np.frombuffer(
            head_depths, dtype=np.float32).reshape((n_frames, size[0], size[1]))
    head_depth = head_depths[idx] * 1000

    pts, colors = backproject_with_color(
        head_depth, 
        head_color_image,
        HEAD_CAM_INTRINSICS,
        mask=head_depth < 3.0,
        NOCS_convention=False
        )
    head_cam_pointcloud = visualize_points(pts, colors)

    # Load labels
    with open('{}/labels.json'.format(data_path), 'r') as f:
        label_dict = json.load(f)
    
    # Prepare state
    label = label_dict[str(idx)]
    ee_cam_pose = np.array(label['ee_cam_pose'])
    ee_cam_pos = ee_cam_pose[:3, 3]
    ee_cam_quat = tra.Rotation.from_matrix(ee_cam_pose[:3, :3]).as_quat()
    head_cam_pose = np.array(label['head_cam_pose'])

    ee_pose = np.array(label['ee_goal_pose'])
    ee_pos = ee_pose[:3, 3]
    ee_quat = tra.Rotation.from_matrix(ee_pose[:3, :3]).as_quat()
    
    # following is the state format for relative motion policy
    if relative_motion:
        current_state = np.array([
            ee_cam_pos[0], ee_cam_pos[1], ee_cam_pos[2],
            ee_cam_quat[0], ee_cam_quat[1], ee_cam_quat[2], ee_cam_quat[3],
            label['gripper']
        ])
    else:
        current_state = np.array([
            ee_pos[0], ee_pos[1], ee_pos[2],
            ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3],
            label['observations']['gripper_finger_right']
        ])
    state = torch.from_numpy(current_state)
    state = state.to(torch.float32)
    state = state.to(device, non_blocking=True)
    state = state.unsqueeze(0)
    
    # Load ground truth actions for comparison
    gt_ee_pose_list = []
    for i in range(chunk_size//4):
        gt_ee_pose = label_dict[str(idx + i*4)]['ee_pose']
        gt_ee_pose_list.append(np.array(gt_ee_pose))
    
    head_cam_pointcloud_inbase = o3d.geometry.PointCloud(head_cam_pointcloud)
    head_cam_pointcloud_inbase.transform(head_cam_pose)
    visualization_data = {
        "ee_cam_pose": ee_cam_pose,
        "head_cam_pointcloud_inbase":  head_cam_pointcloud_inbase,
    }
    
    return state, gripper_color_image_tensor, head_color_image_tensor, gt_ee_pose_list, visualization_data