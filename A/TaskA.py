import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from medmnist import BreastMNIST

# Load data 
training_dataset = BreastMNIST(split="train", download=True)
validation_dataset = BreastMNIST(split="val", download=True)


def dataset_to_arrays(dataset): #Convert MedMNIST dataset to numpy arrays (flattened images and labels)
    images = []
    labels = []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        # Normalize image to [0, 1] and flatten
        if hasattr(image, 'numpy'):
                image = image.numpy()
        elif not isinstance(image, np.ndarray):
                image = np.array(image)
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        images.append(image.flatten())
        
        # Get label value
        label_value = int(label[0]) if isinstance(label, np.ndarray) else int(label)
        labels.append(label_value)
    
    return np.vstack(images), np.array(labels)


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, 
                        min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                        random_state=0):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight='balanced',  # Handle class imbalance
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    
    model.fit(X_train, y_train)
    
    # Evaluate on training data
    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"  Training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    return model


def evaluate_model(model, X, y, dataset_name="Dataset"):
    preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    print(f"\n{dataset_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Evaluated on {len(y)} samples")
    return accuracy


# Main execution
if __name__ == "__main__":
    
    # Convert datasets to numpy arrays
    X_train, y_train = dataset_to_arrays(training_dataset)
    X_val, y_val = dataset_to_arrays(validation_dataset)
    
    
    # Train Random Forest model with strong regularization to reduce overfitting
    model = train_random_forest(
        X_train, y_train,
        n_estimators=100,
        max_depth=10,  # Shallow trees to prevent overfitting
        min_samples_split=30,  
        min_samples_leaf=15,  
        random_state=0
    )

    # Evaluate on training set for comparison
    train_acc = evaluate_model(model, X_train, y_train, "Training")
    
    # Evaluate on validation set
    val_acc = evaluate_model(model, X_val, y_val, "Validation")
    
    # Summary
    gap = train_acc - val_acc
    print("Final Results:")
    print(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Gap (Train - Val):   {gap:.4f} ({gap*100:.2f}%)")