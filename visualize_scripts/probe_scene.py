import pandas as pd
import numpy as np
import open3d as o3d
import cv2
import numpy as np
import cv2
from utils.viewer_utils import SceneViewer
import open3d as o3d
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

T_z_m90 = np.array(
    [
        [0, -1, 0, 0],  # cos(90), -sin(90), 0, 0
        [1, 0, 0, 0],  # sin(90),  cos(90), 0, 0
        [0, 0, 1, 0],  # 0,        0,       1, 0
        [0, 0, 0, 1],  # 0,        0,       0, 1
    ]
).T



def transform_points(points, T):
    points = points @ T[:3, :3].T + T[:3, 3]
    return points


def get_heatmap(values, cmap_name="turbo", invert=False):
    if invert:
        values = -values
    values = (values - values.min()) / (values.max() - values.min())
    colormaps = plt.cm.get_cmap(cmap_name)
    rgb = colormaps(values)[..., :3]  # don't need alpha channel
    return rgb


def visualize_sphere_o3d(center, color=[1, 0, 0], size=0.03):
    # center
    center_o3d = o3d.geometry.TriangleMesh.create_sphere()
    center_o3d.compute_vertex_normals()
    center_o3d.scale(size, [0, 0, 0])
    center_o3d.translate(center)
    center_o3d.paint_uniform_color(color)
    return center_o3d


def visualize_6d_trajectory(
    viewpoints_trajectory,
    size=0.03,
    cmap_name="plasma",
    invert=False,
    primitive="axis",
    to_mesh=False,
    color=None,
):
    vis_o3d = []
    traj_color = get_heatmap(
        np.arange(len(viewpoints_trajectory)), cmap_name=cmap_name, invert=invert
    )
    if color is not None:
        traj_color = np.array(color)[None, :].repeat(len(viewpoints_trajectory), axis=0)
    else:
        traj_color = get_heatmap(
            np.arange(len(viewpoints_trajectory)), cmap_name=cmap_name, invert=invert
        )
    for i, traj_point in enumerate(viewpoints_trajectory):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 10 / 2)
        sphere = visualize_sphere_o3d(
            traj_point[:3, 3], color=traj_color[i], size=size
        )
        axis.transform(traj_point)
        axis += sphere
        vis_o3d.append(axis)
    if to_mesh:
        mesh_o3d = o3d.geometry.TriangleMesh()
        for m in vis_o3d:
            mesh_o3d += m
        return mesh_o3d
    else:
        return vis_o3d


def visualize_points(points, colors=None, as_spheres=False, size=0.02):
    if as_spheres:
        pcd = o3d.geometry.TriangleMesh()
        for i, point in enumerate(points):
            color = colors[i] if colors is not None else [0.5, 0.5, 0.5]
            pcd += visualize_sphere_o3d(point, color=color, size=size)
        return pcd
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def backproject(depth, intrinsics, instance_mask, NOCS_convention=False):
    intrinsics_inv = np.linalg.inv(intrinsics)
    # image_shape = depth.shape
    # width = image_shape[1]
    # height = image_shape[0]

    # x = np.arange(width)
    # y = np.arange(height)

    # non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = depth > 0
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    if NOCS_convention:
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]

    return pts, idxs


# scene_dir = (
#     "/home/wiss/chenh/storage/group/srl/stretch/vidbot_dataset/2026-01-13--16-01-01"
# )
scene_dir = (
    "/home/wiss/chenh/storage/group/srl/stretch/vidbot_dataset/2026-01-13--16-05-43"
)

# Parse the files
depth_files = glob.glob(os.path.join(scene_dir, "depths", "*.png"))
rgb_files = glob.glob(os.path.join(scene_dir, "rgb", "*.png"))
action_files = glob.glob(os.path.join(scene_dir, "dex_traj", "*.npz"))
extr_files = glob.glob(os.path.join(scene_dir, "extr_cam0cam", "*.npz"))
intr_data = np.load(os.path.join(scene_dir, "intr", "intrinsics.npz"))
intr_head, intr_hand = intr_data["HEAD_CAM_K"], intr_data["EE_CAM_K"]

# Sort the files
depth_files.sort()
rgb_files.sort()
action_files.sort()
extr_files.sort()

# Build rgb_frames, depth_frames, intr_frames
colors = [cv2.imread(rgb_file)[..., ::-1].copy() for rgb_file in rgb_files]
depths = [cv2.imread(depth_file, -1) / 1000.0 for depth_file in depth_files]
actions = [np.load(action_file)["trajectory"] for action_file in action_files]
T_wc_list = [np.load(extr_file)["T_base_headcam"] for extr_file in extr_files]
T_wc0 = T_wc_list[0]

vis_scenes, vis_actions, slam_poses = [], [], []

for i in range(len(depth_files)):
    color = colors[i]
    depth = depths[i]
    T_wc = T_wc_list[i]
    action_world = actions[i] # [H, 4, 4]
    T_c0c = np.linalg.inv(T_wc0) @ T_wc
    action_cam0 = np.matmul(np.linalg.inv(T_wc0), action_world) # T_c0w @ T_wa

    # Acquire scene points
    points, scene_ids = backproject(depth, intr_head, depth < 2)
    points_cam0 = transform_points(points, T_c0c)
    point_colors = color[scene_ids[0], scene_ids[1]] / 255.0

    # Do visualization
    pcd_scene = visualize_points(points_cam0, point_colors)
    vis_action = visualize_6d_trajectory(action_cam0, size=0.02, cmap_name="turbo", to_mesh=True)
    vis_scenes.append(pcd_scene)
    vis_actions.append(vis_action)
    slam_poses.append(T_c0c @ T_z_m90.T)

viewer = SceneViewer["o3d"](
    vis_scenes=vis_scenes,
    vis_trajs=vis_actions,
    slam_poses=slam_poses,
    viewer_name="Policy Closed Loop",
    front_distance=1.5,
)
viewer.run()
