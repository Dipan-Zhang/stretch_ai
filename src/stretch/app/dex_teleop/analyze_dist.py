# A script to analyze the collected data by overlapping the first frame of the recordings.
# Fixed: Handles missing/corrupt images gracefully, checks for empty directory, consistent shapes, and cleans up windows.

import os
import cv2
import numpy as np

dataset_path = "/home/chenh/anran_ws/stretch_ai/data/pnp_basketball/default_user/default_env"
camera_name = "head"

images = []
first_shape = None

for episode in sorted(os.listdir(dataset_path)):
    episode_path = os.path.join(dataset_path, episode)
    if not os.path.isdir(episode_path):
        continue
    frame_path = os.path.join(episode_path, f'compressed_{camera_name}_images', '000000.png')
    if not os.path.isfile(frame_path):
        print(f"[WARN] Frame not found: {frame_path}")
        continue
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"[WARN] Could not read image: {frame_path}")
        continue
    if first_shape is None:
        first_shape = frame.shape
    elif frame.shape != first_shape:
        # Resize to match the first image shape.
        frame = cv2.resize(frame, (first_shape[1], first_shape[0]))
    images.append(frame)

if not images:
    print("[ERROR] No images found. Exiting.")
    exit(1)

# Overlap the images
# The original method blends the images sequentially, which results in only the latest two images having a strong effect,
# not averaging over all images. Instead, use the mean of all images for true averaging/overlap visualization.
combined_image = np.mean([img.astype(np.float32) for img in images], axis=0)

# Convert back to uint8
combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

# Display & save image
cv2.imshow("Overlapped Images", combined_image)
# cv2.imwrite("overlapped_first_frames.png", combined_image)
print("Image saved as overlapped_first_frames.png")
cv2.waitKey(0)
cv2.destroyAllWindows()