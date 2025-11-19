import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from medmnist import BreastMNIST

# Load data
dataset = training_dataset = BreastMNIST(split="train", download=True, size=64)
validation_dataset = BreastMNIST(split="val", download=True, size=64)


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
                image = image.squeeze(0)  
            ax.imshow(image, cmap='grey')  # Greyscale

            label_val = label.item() if hasattr(label, 'item') else int(label)
            class_name = class_names.get(str(label_val), f"Class {label_val}")
            ax.set_title(f"Label: {class_name}\n({label_val})", fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle('BreastMNIST - Sample Images', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def dataset_to_arrays(dataset):
    images, labels = [], []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        image = np.array(image, dtype=np.float32) / 255.0  # normalisation
        images.append(image.flatten())
        label_value = int(label[0]) if isinstance(label, np.ndarray) else int(label)
        labels.append(label_value)
    return np.vstack(images), np.array(labels)


def train_linear_svc(dataset):
    X, y = dataset_to_arrays(dataset)
    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=546,
        dual=False,
        random_state=0,
    )
    model.fit(X, y)
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    print(f"\nTrained Linear SVC on {len(y)} samples.")
    print(f"Training accuracy (same data): {accuracy:.4f}")
    return model

# Testing on validation dataset
def evaluate_model(model, validation_dataset):
    X_val, y_val = dataset_to_arrays(validation_dataset)
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print(f"Evaluated on {len(y_val)} validation samples.")
    return accuracy


# Output information about loaded data
if __name__ == "__main__":
    print(f"Training dataset size: {len(dataset)} samples")
    print(f"  Description: {dataset.info.get('description', 'N/A')}")
    print(f"  Task type: {dataset.info.get('task', 'N/A')}")
    print(f"  Number of channels: {dataset.info.get('n_channels', 'N/A')}")
    print(f"  Classes: {dataset.info.get('label', {})}")
    print(f"  Image size: 64x64 pixels")
    print(f"\nData split: Validation set")
    print(f"Total samples in validation set: {len(validation_dataset)}")
    
    # Display sample images
    #print("\nDisplaying sample images...")
    #display_sample_images(dataset, num_samples=8)

    #print("\nFitting Linear SVC model on the training dataset...")
    model = train_linear_svc(training_dataset)
    
    print("\nEvaluating model on validation dataset...")
    evaluate_model(model, validation_dataset)
