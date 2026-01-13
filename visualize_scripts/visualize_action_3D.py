import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import os
import json
# import viser
import open3d as o3d  
import scipy.spatial.transform as tra
from stretch.motion.kinematics import HelloStretchKinematics, HelloStretchIdx
from stretch.app.lfd.visualize_utils import backproject_with_color, visualize_points
import liblzfse
# def add_frame(server, name, T):
#     position = T[:3, 3]
#     # scipy Rotation.as_quat does not support scalar_first; returns [x, y, z, w]
#     xyzw = tra.Rotation.from_matrix(T[:3, :3]).as_quat()
#     # Reorder to w, x, y, z (scalar first) for libraries that expect that format
#     wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
#     server.scene.add_frame(
#         name,
#         wxyz=wxyz,
#         position=position,
#     )

# for vidbot
HEAD_CAM_K= np.array([[911.0936279296875, 0.0, 377.4052734375], [0.0, 910.6778564453125, 626.5808715820312], [0.0, 0.0, 1.0]])
EE_CAM_K = np.array([[431.0948181152344, 0.0, 313.27081298828125], [0.0, 430.81134033203125, 245.1189422607422], [0.0, 0.0, 1.0]])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_idx', type=int, default=10)
    parser.add_argument('--future_steps', type=int, default=8)
    parser.add_argument('--camera_name', type=str, default='head')
    args = parser.parse_args()
    frame_idx = args.frame_idx
    future_steps = args.future_steps
    camera_name = args.camera_name
    
    # DATA_PATH = '/home/wiss/zanr/code/stretch/data/pickup_bottle_v2/default_user/default_env/'
    DATA_PATH = '/home/chenh/hanzhi_ws/egoprior-diffuser/data/task/default_user/default_env/'
    selected_folder = os.path.join(DATA_PATH, f'2026-01-13--16-05-43')
    datafile = os.path.join(selected_folder, 'labels_interpolated.json')
    with open(datafile, 'r') as f:
        recordings_dict = json.load(f)
    assert str(frame_idx) in recordings_dict, f"Frame {frame_idx} not found in recordings_dict"

    # Load camera image
    camera_fn = os.path.join(selected_folder, f'compressed_{camera_name}_images', f'{frame_idx:06d}.png')
    if not os.path.exists(camera_fn):
        raise FileNotFoundError(f"Camera image not found: {camera_fn}")
    else:
        head_img = cv2.imread(camera_fn, -1)
        head_img = cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB)
    
    n_frames = len(os.listdir(os.path.join(selected_folder, f'compressed_head_images')))
    head_depth_fname= os.path.join(selected_folder, f'compressed_np_head_depth_float32.bin')
    size = head_img.shape[:2]
    with open(head_depth_fname, 'rb') as f:
        head_depths = liblzfse.decompress(f.read())
        head_depths = np.frombuffer(
            head_depths, dtype=np.float32).reshape((n_frames, size[0], size[1]))
    head_depth = head_depths[frame_idx] 

    # visualize current scene
    pts, colors = backproject_with_color(
        head_depth, 
        head_img,
        HEAD_CAM_K,
        mask=head_depth < 3.0,
        NOCS_convention=False
    )

    scene_pcd = visualize_points(pts, colors)

    # load the future actions
    actions_in_base = []
    actions_in_head = []
    states_in_head = []
    states_in_base = []

    T_base_head_cami = np.array(recordings_dict[str(frame_idx)]['head_cam_pose'])

    # T_base_head_cam0 = np.array(recordings_dict[str(0)]['head_cam_pose'])
    for i in range(future_steps):
        action = np.array(recordings_dict[str(frame_idx + i)]['ee_goal_pose'])
        state = np.array(recordings_dict[str(frame_idx + i)]['observations']['ee_pose'])

        # print(action@action.transpose()==np.eye(4))
        cam_pose_temp = np.array(recordings_dict[str(frame_idx+i)]['head_cam_pose'])
        # print(np.allclose(cam_pose_temp@np.linalg.inv(cam_pose_temp), np.eye(4), atol=1e-3))
        actions_in_base.append(action)
        states_in_base.append(state)

        action_in_head = np.linalg.inv(T_base_head_cami) @ action
        action_state_in_head =  np.linalg.inv(T_base_head_cami) @ state
        actions_in_head.append(action_in_head)
        states_in_head.append(action_state_in_head)


    # visualize the future actions
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
    actions_in_head_vis = []
    for action in actions_in_head:
        action_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        action_vis.transform(action)
        actions_in_head_vis.append(action_vis)
    o3d.visualization.draw_geometries([scene_pcd, world_frame] + actions_in_head_vis)

    # visualize the future states
    states_in_head_vis = []
    for state in states_in_head:
        state_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        state_vis.transform(state)
        states_in_head_vis.append(state_vis)
    o3d.visualization.draw_geometries([scene_pcd, world_frame] + states_in_head_vis)






