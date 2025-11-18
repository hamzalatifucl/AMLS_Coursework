import numpy as np
import matplotlib.pyplot as plt
from medmnist import BreastMNIST

# Load data
dataset = BreastMNIST(split="val", download=True, size=64)


def display_sample_images(dataset, num_samples=8):
    class_names = dataset.info.get('label', {})
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    for idx, ax in enumerate(axes):
        if idx < len(indices):
            sample_idx = indices[idx]
            image, label = dataset[sample_idx]
            if hasattr(image, 'numpy'):
                image = image.numpy()
            elif not isinstance(image, np.ndarray):
                image = np.array(image)
            if len(image.shape) == 3:
                if image.shape[0] == 1:  # Grayscale with channel dimension at start
                    image = image.squeeze(0)
                elif image.shape[2] == 1:  # Grayscale with channel at end
                    image = image.squeeze(2)
            if len(image.shape) == 2:  
                ax.imshow(image, cmap='grey') # Grayscale
            else:  # RGB
                ax.imshow(image)

            label_val = label.item() if hasattr(label, 'item') else int(label)
            class_name = class_names.get(str(label_val), f"Class {label_val}")
            ax.set_title(f"Label: {class_name}\n({label_val})", fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    plt.suptitle('BreastMNIST - Sample Images', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# Output information about loaded data
if __name__ == "__main__":
    print("Task A: BreastMNIST Dataset")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"  Description: {dataset.info.get('description', 'N/A')}")
    print(f"  Task type: {dataset.info.get('task', 'N/A')}")
    print(f"  Number of channels: {dataset.info.get('n_channels', 'N/A')}")
    print(f"  Classes: {dataset.info.get('label', {})}")
    print(f"  Image size: 64x64 pixels")
    print(f"\nData split: Validation set")
    print(f"Total samples in validation set: {len(dataset)}")
    
    # Display sample images
    print("\nDisplaying sample images...")
    display_sample_images(dataset, num_samples=8)
