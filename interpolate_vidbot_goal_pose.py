import json
import numpy as np
import os

data_path = '/home/chenh/hanzhi_ws/egoprior-diffuser/data/task/default_user/default_env/'
all_data_fn = os.listdir(data_path)
selected_data_fn = [f for f in all_data_fn if '2026-01-13' in f]

for file_name in selected_data_fn:
    # read the data from the json file
    datafile = os.path.join(data_path, file_name, 'labels.json')
    with open(datafile, 'r') as f:
        data = json.load(f)

    # Assumes: each 'ee_goal_pose' and 'ee_pose' is a 4x4 SE(3), stored as a nested list.

    keys_sorted = sorted(data.keys(), key=lambda x: int(x))  # numerical order

    # Deep copy for interpolation
    data_interpolated = json.loads(json.dumps(data))

    # Find all segments where goal pose stays constant
    segments = []
    current_segment_start = 0
    current_goal_pos = None

    for idx, k in enumerate(keys_sorted):
        frame = data[k]
        curr_goal_pose = np.array(frame['ee_goal_pose'])
        curr_goal_pos = curr_goal_pose[:3, 3]
        
        if current_goal_pos is None:
            # First frame
            current_goal_pos = curr_goal_pos
            current_segment_start = idx
        elif not np.allclose(curr_goal_pos, current_goal_pos):
            # Goal changed, save previous segment
            segments.append((current_segment_start, idx - 1, current_goal_pos))
            current_goal_pos = curr_goal_pos
            current_segment_start = idx

    # Add the last segment
    segments.append((current_segment_start, len(keys_sorted) - 1, current_goal_pos))

    print(f'Found {len(segments)} goal pose segments')

    First_seg = True
    # Interpolate each segment
    for seg_idx, (start_idx, end_idx, target_goal_pos) in enumerate(segments):
        # Get the current pose at the start of this segment
        start_key = keys_sorted[start_idx]
        start_frame = data[start_key]
        if First_seg: 
            start_pose = np.array(start_frame['ee_pose'])
            First_seg = False
        else:
            # start interpolation from last segment's end pose
            start_pose = np.array(data[keys_sorted[start_idx-1]]['ee_goal_pose'])
        start_pos = start_pose[:3, 3]
        
        # Get the target goal position for this segment
        target_pos = target_goal_pos
        
        # Number of steps in this segment
        num_steps = end_idx - start_idx + 1
        
        print(f'Segment {seg_idx + 1}: steps {start_idx} to {end_idx}, '
            f'interpolating from {start_pos} to {target_pos}')
        
        # Interpolate positions for all frames in this segment
        for i in range(num_steps):
            frame_idx = start_idx + i
            key = keys_sorted[frame_idx]
            
            # Linear interpolation: alpha goes from 0 to 1
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
            new_pos = (1 - alpha) * start_pos + alpha * target_pos
            
            # Update the goal pose
            ee_goal_pose = np.array(data_interpolated[key]['ee_goal_pose'])
            ee_goal_pose[:3, 3] = new_pos
            data_interpolated[key]['ee_goal_pose'] = ee_goal_pose.tolist()

    # save the new trajectory to a json file
    new_datafile = datafile.replace('.json', '_interpolated.json')
    with open(new_datafile, 'w') as f:
        json.dump(data_interpolated, f, indent=2)

    print(f'Interpolated trajectory saved to: {new_datafile}')