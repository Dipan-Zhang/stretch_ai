import numpy as np
import cv2

def process_vertical_image(orig_image: np.ndarray, target_height: int, target_width: int, intrinsic: np.ndarray = None, cut_mode: str = "top"):
    """
    process the vertical image and return the new image and intrinsic matrix
    Args:
        orig_image: np.ndarray, the original image
        target_height: int, the target height of the cropped image
        target_width: int, the target width of the cropped image
        intrinsic: np.ndarray, the intrinsic matrix of the camera
        cut_mode: str, the mode to crop the image
    Returns:
        new_image: np.ndarray, the new image
        new_intrinsic: np.ndarray, the new intrinsic matrix
    """
    if orig_image.ndim == 2:
        DEPTH_MODE=True
    else:
        DEPTH_MODE=False
    orig_height, orig_width = orig_image.shape[0], orig_image.shape[1]
    assert orig_height > orig_width, "Original height must be greater than width"
    

    # 1. Determine Crop Offset
    cropped_height = orig_width
    padding = (orig_height - cropped_height) // 2
    if cut_mode == "bottom":
        y_offset = 2 * padding
    elif cut_mode == "center":
        y_offset = padding
    elif cut_mode == "top":
        y_offset = 0
    else:
        raise NotImplementedError
    
    # apply resize 
    if DEPTH_MODE:
        new_image = np.zeros((orig_width, orig_width), dtype=np.float32)
        new_image = orig_image[y_offset : y_offset + orig_width, 0 : orig_width]
        new_image_resized = cv2.resize(new_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    else:
        new_image = np.zeros((orig_width, orig_width, 3), dtype=np.uint8)
        new_image = orig_image[y_offset : y_offset + orig_width, 0 : orig_width, :]
        new_image_resized = cv2.resize(new_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        
    # 2. Update Intrinsic for Crop
    if intrinsic is not None:
        new_intrinsic = intrinsic.copy()
        # Shift principal point by the crop offset
        # cx remains same because x_offset is 0
        new_intrinsic[1, 2] = intrinsic[1, 2] - y_offset 
        
        # 3. Handle Resize
        # Note: new_image currently has shape (target_height, orig_width)
        # We are resizing it to (target_width, target_height)
        scale_x = target_width / orig_width
        scale_y = target_height / cropped_height # This is 1.0 in your current logic!
        
        # Apply scaling to the whole matrix (fx, fy, cx, cy)
        new_intrinsic[0, 0] *= scale_x
        new_intrinsic[0, 2] *= scale_x
        new_intrinsic[1, 1] *= scale_y
        new_intrinsic[1, 2] *= scale_y
    else:
        new_intrinsic = None

    return new_image_resized, new_intrinsic