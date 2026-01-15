import numpy as np
from scipy.ndimage import rotate, gaussian_filter


def augment_image(image_2d, rotation_range=3, blur_sigma=0.3):
    """Apply random augmentations to image"""
    img = image_2d.copy()
    
    # Random rotation 
    if rotation_range > 0 and np.random.random() > 0.7:
        angle = np.random.uniform(-rotation_range, rotation_range)
        img = rotate(img, angle, axes=(0, 1), reshape=False, mode='constant', cval=0.0)
    
    # Blur
    if blur_sigma > 0 and np.random.random() > 0.7:
        img = gaussian_filter(img, sigma=blur_sigma)
    
    return img


def dataset_to_arrays(dataset, apply_augmentation=False):
    """doubling the dataset size with augmented images"""
    images, labels = [], []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        if hasattr(image, 'numpy'):
            image = image.numpy()
        elif not isinstance(image, np.ndarray):
            image = np.array(image)
        image = image.astype(np.float32) / 255.0
        label_value = int(label[0]) if isinstance(label, np.ndarray) else int(label)
        image_2d = image[:, :, 0] if len(image.shape) == 3 else image
        
        # Keep images as 2D (untouched for raw data)
        images.append(image_2d)
        labels.append(label_value)
        
        if apply_augmentation:
            augmented = augment_image(image_2d)
            images.append(augmented)
            labels.append(label_value)
    
    return np.array(images), np.array(labels)

