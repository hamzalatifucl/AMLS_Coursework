import numpy as np
import torch
from torchvision.transforms import functional as F


def apply_random_rotation(img, max_angle=3):
    """Apply random rotation"""
    if np.random.random() > 0.7:
        angle = np.random.uniform(-max_angle, max_angle)
        rotated = F.rotate(img, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0)
        if len(rotated.shape) != 3 or rotated.shape[0] != 1:
            if len(rotated.shape) == 2:
                rotated = rotated.unsqueeze(0)
        return rotated
    return img


def apply_blur(img, sigma=0.3):
    """Apply blur"""
    if np.random.random() > 0.7:
        from scipy.ndimage import gaussian_filter
        # Get device before converting to numpy
        device = img.device
        # Convert to numpy (handle both 3D and 4D)
        if len(img.shape) == 4:
            img_np = img.squeeze(0).cpu().detach().numpy()
        else:
            img_np = img.cpu().detach().numpy()
        
        # Apply blur based on shape
        if len(img_np.shape) == 3:  # (C, H, W)
            img_blurred = np.array([gaussian_filter(img_np[i], sigma=sigma) for i in range(img_np.shape[0])])
        elif len(img_np.shape) == 2:  # (H, W)
            img_blurred = gaussian_filter(img_np, sigma=sigma)
            img_blurred = img_blurred[np.newaxis, ...]  # Add channel dimension
        else:
            img_blurred = img_np
        
        # Convert back to tensor and ensure correct shape
        result = torch.FloatTensor(img_blurred).to(device)
        # Ensure output shape matches input exactly
        if len(img.shape) == 4 and len(result.shape) == 3:
            result = result.unsqueeze(0)
        elif len(img.shape) == 3 and len(result.shape) == 2:
            result = result.unsqueeze(0)
        return result
    return img


def dynamic_augment(img):
    """Apply random augmentations of rotation and blur"""
    img = apply_random_rotation(img)
    img = apply_blur(img)
    return img

