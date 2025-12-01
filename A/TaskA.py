import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from medmnist import BreastMNIST

# Load data
dataset = training_dataset = BreastMNIST(split="train", download=True, size=64)
validation_dataset = BreastMNIST(split="val", download=True, size=64)


def dataset_to_arrays(dataset):
    images, labels = [], []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        image = np.array(image, dtype=np.float32) / 255.0  # normalisation
        images.append(image.flatten())
        label_value = int(label[0]) if isinstance(label, np.ndarray) else int(label)
        labels.append(label_value)
    return np.vstack(images), np.array(labels)


def train_linear_svc(dataset, C=1.0, max_iter=10000):
    """Train a Linear SVC model with configurable hyperparameters."""
    X, y = dataset_to_arrays(dataset)
    model = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        dual=False,
        random_state=0,
    )
    model.fit(X, y)
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    print(f"\nTrained Linear SVC on {len(y)} samples.")
    print(f"  C={C}, max_iter={max_iter}")
    print(f"Training accuracy (same data): {accuracy:.4f}")
    return model


def evaluate_model(model, validation_dataset):
    """Evaluate model on validation dataset."""
    X_val, y_val = dataset_to_arrays(validation_dataset)
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print(f"Evaluated on {len(y_val)} validation samples.")
    return accuracy


def train_with_pca(training_dataset, validation_dataset, n_components=151, C=0.2):
    """Train model with PCA dimensionality reduction."""
    X_train, y_train = dataset_to_arrays(training_dataset)
    X_val, y_val = dataset_to_arrays(validation_dataset)
    
    # Standardize features before PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Find how many components are needed for 95% variance
    # print("Finding Components Needed for 95% Variance")
    # pca_full = PCA()
    # pca_full.fit(X_train_scaled)
    # cumsum = pca_full.explained_variance_ratio_.cumsum()
    # d = np.argmax(cumsum >= 0.95) + 1
    # print(f"You need {d} components to keep 95% of the image details.")
    
    # PCA with 151 components
    pca = PCA(151)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"\nUsing 151 PCA components")
    print(f"Explained variance: {explained_var*100:.2f}%")
    print(f"Reduced from 4096 to 151 features")
    
    # Train model
    model = LinearSVC(C=C, class_weight="balanced", max_iter=10000, dual=False, random_state=0)
    model.fit(X_train_pca, y_train)
    
    # Evaluate model
    train_preds = model.predict(X_train_pca)
    val_preds = model.predict(X_val_pca)
    
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    gap = train_acc - val_acc
    
    print(f"\nPCA Model Results:")
    print(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Gap:                 {gap:.4f} ({gap*100:.2f}%)")
    
    return model, scaler, pca, train_acc, val_acc, gap


# Main execution
if __name__ == "__main__":
    # print(f"Training dataset size: {len(dataset)} samples")
    # print(f"  Description: {dataset.info.get('description', 'N/A')}")
    # print(f"  Task type: {dataset.info.get('task', 'N/A')}")
    # print(f"  Number of channels: {dataset.info.get('n_channels', 'N/A')}")
    # print(f"  Classes: {dataset.info.get('label', {})}")
    # print(f"  Image size: 64x64 pixels")
    # print(f"\nData split: Validation set")
    # print(f"Total samples in validation set: {len(validation_dataset)}")
    
    # Train original model (4096 features)
    print("Training Original Model (4096 features)")
    model_original = train_linear_svc(training_dataset, C=1.0, max_iter=10000)
    original_val_acc = evaluate_model(model_original, validation_dataset)
    
    # Train model with PCA (151 components)
    print("Training Model with PCA (151 components)")
    model_pca, scaler, pca, pca_train_acc, pca_val_acc, pca_gap = train_with_pca(
        training_dataset, validation_dataset, n_components=151, C=0.2
    )
    
    # Compare original vs PCA
    print("Comparison: Original vs PCA")
    print(f"{'Model':<30} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<10}")
    print(f"{'Original (4096 features)':<30} {1.0000:<12.4f} {original_val_acc:<12.4f} {1.0000 - original_val_acc:<10.4f}")
    print(f"{'PCA (151 components)':<30} {pca_train_acc:<12.4f} {pca_val_acc:<12.4f} {pca_gap:<10.4f}")
    

