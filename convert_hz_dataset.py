# load the dataset from saved dataset
import os
import json
import numpy
import cv2
# import viser
import open3d as o3d  
import scipy.spatial.transform as tra
import liblzfse
import numpy as np


HEAD_CAM_K= np.array([[911.0936279296875, 0.0, 377.4052734375], [0.0, 910.6778564453125, 626.5808715820312], [0.0, 0.0, 1.0]])
EE_CAM_K = np.array([[431.0948181152344, 0.0, 313.27081298828125], [0.0, 430.81134033203125, 245.1189422607422], [0.0, 0.0, 1.0]])

original_dataset_path = '/home/wiss/zanr/code/stretch/data/recordings_vidbot/'
file_names = ['2026-01-13--16-05-43', '2026-01-13--16-01-01']




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--future_steps', type=int, default=8)
    parser.add_argument('--camera_name', type=str, default='head')
    args = parser.parse_args()
    future_steps = args.future_steps
    camera_name = args.camera_name


    for file_name in file_names:
        dataset_path = os.path.join(original_dataset_path, file_name)

        target_dir = '/home/wiss/zanr/storage/group/srl/stretch/vidbot_dataset'
        target_save_dir = os.path.join(target_dir, file_name)
        os.makedirs(target_save_dir, exist_ok=True)
        
        datafile = os.path.join(dataset_path, 'labels_interpolated.json')
        with open(datafile, 'r') as f:
            recordings_dict = json.load(f)
        num_frames = len(recordings_dict)

        frame_indices_to_be_converted = list(range(0, num_frames - future_steps))

        # save intrinsics
        intrinsics_dir = os.path.join(target_save_dir, 'intr')
        os.makedirs(intrinsics_dir, exist_ok=True)
        np.savez(os.path.join(intrinsics_dir, 'intrinsics.npz'), HEAD_CAM_K=HEAD_CAM_K, EE_CAM_K=EE_CAM_K)

        for frame_idx in frame_indices_to_be_converted:
            # Load camera image
            camera_fn = os.path.join(dataset_path, f'compressed_{camera_name}_images', f'{frame_idx:06d}.png')
            if not os.path.exists(camera_fn):
                raise FileNotFoundError(f"Camera image not found: {camera_fn}")
            else:
                head_img = cv2.imread(camera_fn, -1)
                head_img = cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB)
            
            T_base_headcam = np.array(recordings_dict[str(frame_idx)]['head_cam_pose'])
            T_base_eecam = np.array(recordings_dict[str(frame_idx)]['ee_cam_pose'])
            
            # load depth
            n_frames = len(os.listdir(os.path.join(dataset_path, f'compressed_head_images')))
            head_depth_fname= os.path.join(dataset_path, f'compressed_np_head_depth_float32.bin')
            size = head_img.shape[:2]
            with open(head_depth_fname, 'rb') as f:
                head_depths = liblzfse.decompress(f.read())
                head_depths = np.frombuffer(
                    head_depths, dtype=np.float32).reshape((n_frames, size[0], size[1]))
            head_depth = head_depths[frame_idx] 

        
            # load action trajectories
            actions_in_base = []
            for i in range(future_steps):
                action = recordings_dict[str(frame_idx + i)]['observations']['ee_pose']
                actions_in_base.append(action)
            actions_in_base = np.array(actions_in_base)
            
            # save images
            head_img = cv2.cvtColor(head_img, cv2.COLOR_RGB2BGR)
            image_dir = os.path.join(target_save_dir, 'rgb')
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{frame_idx:06d}.png')
            cv2.imwrite(image_path, head_img)

            # save depth
            head_depth = (head_depth * 1000).astype(np.uint16)
            depth_dir = os.path.join(target_save_dir, 'depth')
            os.makedirs(depth_dir, exist_ok=True)
            depth_path = os.path.join(depth_dir, f'{frame_idx:06d}.png')
            cv2.imwrite(depth_path, head_depth)

            # save action trajectories
            dex_traj_dir = os.path.join(target_save_dir, 'dex_traj')
            os.makedirs(dex_traj_dir, exist_ok=True)
            save_fn_actions = os.path.join(dex_traj_dir, f'{frame_idx:06d}.npz')
            np.savez(save_fn_actions, trajectory=actions_in_base)

            # save extrinsics of camera
            extr_cam0cam_dir = os.path.join(target_save_dir, 'extr_cam0cam')
            os.makedirs(extr_cam0cam_dir, exist_ok=True)
            save_fn_extrinsics = os.path.join(extr_cam0cam_dir, f'{frame_idx:06d}.npz')
            np.savez(save_fn_extrinsics, T_base_headcam=T_base_headcam, T_base_eecam=T_base_eecam)




