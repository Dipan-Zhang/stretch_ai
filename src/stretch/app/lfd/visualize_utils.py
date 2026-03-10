import numpy as np
import open3d as o3d
import scipy.spatial.transform as tra
import json
import cv2
import os
import torch
import liblzfse
# import stretch.app.lfd.policy_utils as policy_utils
import matplotlib.pyplot as plt
from typing import Any

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


def project_action_predictions(actions: np.ndarray, T_base_cam: np.ndarray, head_cam_K: np.ndarray, head_cam_img: np.ndarray, cmap_name: str = "turbo"):
    """
    Project the actions (in base frame) to the head image
    Args:
        actions: (N, 8) array of actions in base frame
        T_base_cam: (4, 4) transformation matrix from base to camera frame
        head_cam_img: (H, W, 3) array of head camera image (RGB, uint8)
    Returns:
        img: (H, W, 3) array of projected actions on the head image
    """

    if actions.shape[-1] == 9: 
        # visualize random samples within a bat
        preds_pixels, preds_pts = project_pts_in_cam(
            actions[:,:3], 
            head_cam_K.astype(np.float32), 
            T_base_cam)

        img = visualize_projected_pixels(preds_pixels, head_cam_img, cmap_name=cmap_name)
        return img
    else:
        print("invalid actions shape, should be N, 9")
        return head_cam_img


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

def get_heatmap(values, cmap_name="turbo", invert=False):
    if invert:
        values = -values
    values = (values - values.min()) / (values.max() - values.min())
    colormaps = plt.cm.get_cmap(cmap_name)
    rgb = colormaps(values)[..., :3]  # don't need alpha channel
    return rgb


def visualize_projected_pixels(preds_pixels: list, image: np.ndarray, ee_cam_pixels: list | None = None, cmap_name: str = "turbo") -> np.ndarray:
    """
    Visualize projected pixels on the image using OpenCV.
    Args:
        preds_pixels: (N, 2) array of projected pixels for predictions

        ee_cam_pixels: (N, 2) array of projected pixels for end effector
        image: (H, W, 3) array of image (RGB or BGR)
    Returns:
        np.ndarray: The resulting image with projected points drawn on it.
    """
    # Make a copy so we don't overwrite the original
    img = image.copy()
    # H, W = img.shape[:2]  # Get image height and width
    
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    # ax.imshow(img)

    # alphas = np.linspace(1, 0.2, len(preds_pixels))
    traj_colors = get_heatmap(np.arange(len(preds_pixels))[None], cmap_name, invert=False)[0]
    for i, wp in enumerate(preds_pixels):
        wp_color = (traj_colors[i] * 255).astype(np.uint8)
        wp = np.floor(wp).astype(np.int32)
        radii = 7
        # radii = min_radii + (max_radii - min_radii) * time_norm
        img = cv2.circle(
            img,
            center=(int(wp[0]), int(wp[1])),
            radius=int(radii),
            color=(int(wp_color[0]), int(wp_color[1]), int(wp_color[2])),
            thickness=-1,
        )
    # for i, (u, v) in enumerate[Any](preds_pixels):
    #     # Check if prediction pixel is within image bounds
    #     if 0 <= u < W and 0 <= v < H:
    #         ax.plot(u, v, 'o', markersize=15, label='predictions', color=colors[i])

    # # obs might different length
    # if ee_cam_pixels is not None:   
    #     for i, (u, v) in enumerate(ee_cam_pixels):
    #         # Check if end effector pixel is within image bounds
    #         if 0 <= u < W and 0 <= v < H:
    #             ax.plot(u, v, '*', markersize=15, label='obs_ee_proj', color='blue', alpha=alphas[i])

    # ax.axis('off')
    # fig.tight_layout()
    # # convert to numpy array
    # fig.canvas.draw()
    # projected_img = np.asarray(fig.canvas.buffer_rgba())
    # projected_img = projected_img[:, :, :3]
    # plt.close(fig)
    return img  