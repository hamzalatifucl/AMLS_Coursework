import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from sklearn.metrics import accuracy_score
from medmnist import BreastMNIST

# ============================================================================
# OLD MODEL CODE (COMMENTED OUT)
# ============================================================================
# from sklearn.metrics import accuracy_score
# from sklearn.svm import LinearSVC
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# 
# def dataset_to_arrays(dataset):
#     images, labels = [], []
#     for idx in range(len(dataset)):
#         image, label = dataset[idx]
#         image = np.array(image, dtype=np.float32) / 255.0  # normalisation
#         images.append(image.flatten())
#         label_value = int(label[0]) if isinstance(label, np.ndarray) else int(label)
#         labels.append(label_value)
#     return np.vstack(images), np.array(labels)
# 
# def train_linear_svc(dataset, C=1.0, max_iter=10000):
#     """Train a Linear SVC model with configurable hyperparameters."""
#     X, y = dataset_to_arrays(dataset)
#     model = LinearSVC(
#         C=C,
#         class_weight="balanced",
#         max_iter=max_iter,
#         dual=False,
#         random_state=0,
#     )
#     model.fit(X, y)
#     preds = model.predict(X)
#     accuracy = accuracy_score(y, preds)
#     print(f"\nTrained Linear SVC on {len(y)} samples.")
#     print(f"  C={C}, max_iter={max_iter}")
#     print(f"Training accuracy (same data): {accuracy:.4f}")
#     return model
# 
# def evaluate_model(model, validation_dataset):
#     """Evaluate model on validation dataset."""
#     X_val, y_val = dataset_to_arrays(validation_dataset)
#     preds = model.predict(X_val)
#     accuracy = accuracy_score(y_val, preds)
#     print(f"\nValidation Accuracy: {accuracy:.4f}")
#     print(f"Evaluated on {len(y_val)} validation samples.")
#     return accuracy
# 
# def train_with_pca(training_dataset, validation_dataset, n_components=151, C=0.2):
#     """Train model with PCA dimensionality reduction."""
#     X_train, y_train = dataset_to_arrays(training_dataset)
#     X_val, y_val = dataset_to_arrays(validation_dataset)
#     
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
#     
#     pca = PCA(n_components=n_components)
#     X_train_pca = pca.fit_transform(X_train_scaled)
#     X_val_pca = pca.transform(X_val_scaled)
#     
#     model = LinearSVC(C=C, class_weight="balanced", max_iter=10000, dual=False, random_state=0)
#     model.fit(X_train_pca, y_train)
#     
#     train_preds = model.predict(X_train_pca)
#     val_preds = model.predict(X_val_pca)
#     
#     train_acc = accuracy_score(y_train, train_preds)
#     val_acc = accuracy_score(y_val, val_preds)
#     gap = train_acc - val_acc
#     
#     print(f"\nPCA Model Results:")
#     print(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
#     print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
#     print(f"  Gap:                 {gap:.4f} ({gap*100:.2f}%)")
#     
#     return model, scaler, pca, train_acc, val_acc, gap

# Load data
dataset = training_dataset = BreastMNIST(split="train", download=True, size=64)
validation_dataset = BreastMNIST(split="val", download=True, size=64)



class BreastMNISTDataset(Dataset):
    def __init__(self, medmnist_dataset, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform  # Store transform for dynamic augmentation
        
        for idx in range(len(medmnist_dataset)):
            image, label = medmnist_dataset[idx]
            # normalization with numpy array
            if hasattr(image, 'numpy'):
                image = image.numpy()
            elif not isinstance(image, np.ndarray):
                image = np.array(image)
            image = image.astype(np.float32) / 255.0  
            
            # Get label
            label_value = int(label[0]) if isinstance(label, np.ndarray) else int(label)
            
            self.images.append(image)
            self.labels.append(label_value)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx]).unsqueeze(0)  # added channel dimension
        
        # Apply transform dynamically - each epoch will see different augmentations
        if self.transform is not None:
            image = self.transform(image)
        
        label = torch.LongTensor([self.labels[idx]])[0]  
        return image, label


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNClassifier, self).__init__()
        
        # layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def train_neural_network(train_loader, val_loader, num_epochs=100, learning_rate=0.001, patience=20):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNNClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = [] # entropy losses
    train_accs = []
    val_accs = []
    
    # Best model tracking
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_train_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    stopped_early = False
    
    print(f"\nEarly stopping: Will stop if validation doesn't improve for {patience} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = correct_val / total_val
        val_accs.append(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_train_acc = train_acc  # Save training accuracy at best validation
            best_train_loss = train_loss  # Save training loss at best validation
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
        
        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Validation accuracy did not improve for {patience} epochs")
            stopped_early = True
            # Restore best model
            model.load_state_dict(best_model_state)
            break
    
    # Show metrics from best model
    if not stopped_early:
        print("Training completed all epochs")
        print(f"\nBest model was at epoch {best_epoch}:")
        print(f"  Training Loss:   {best_train_loss:.4f}")
        print(f"  Training Acc:    {best_train_acc:.4f} ({best_train_acc*100:.2f}%)")
        print(f"  Validation Acc:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        model.load_state_dict(best_model_state)
    
    print("Training Complete")
    print(f"✓ Using best model from epoch {best_epoch} for evaluation")
    print(f"  This model will be used on test data")
    print(f"\nBest model metrics at epoch {best_epoch}:")
    print(f"  Training Loss:   {best_train_loss:.4f}")
    print(f"  Training Acc:    {best_train_acc:.4f} ({best_train_acc*100:.2f}%)")
    print(f"  Validation Acc:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    if stopped_early:
        print(f"\nStopped early at epoch {epoch+1} (saved {num_epochs - epoch - 1} epochs)")
    
    return model, train_accs, val_accs


def evaluate_neural_network(model, data_loader):
    device = torch.device('cpu')
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, all_preds, all_labels


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
    


    # OLD MODEL CODE (COMMENTED OUT)

    # # Train original model (4096 features)
    # print("Training Original Model (4096 features)")
    # model_original = train_linear_svc(training_dataset, C=1.0, max_iter=10000)
    # original_val_acc = evaluate_model(model_original, validation_dataset)
    # 
    # # Train model with PCA (151 components)
    # print("Training Model with PCA (151 components)")
    # model_pca, scaler, pca, pca_train_acc, pca_val_acc, pca_gap = train_with_pca(
    #     training_dataset, validation_dataset, n_components=151, C=0.2
    # )
    # 
    # # Compare original vs PCA
    # print("Comparison: Original vs PCA")
    # print(f"{'Model':<30} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<10}")
    # print(f"{'Original (4096 features)':<30} {1.0000:<12.4f} {original_val_acc:<12.4f} {1.0000 - original_val_acc:<10.4f}")
    # print(f"{'PCA (151 components)':<30} {pca_train_acc:<12.4f} {pca_val_acc:<12.4f} {pca_gap:<10.4f}")
    # 
    # improvement = pca_val_acc - original_val_acc
    # gap_reduction = (1.0000 - original_val_acc) - pca_gap
    # print(f"\nValidation Accuracy: {improvement:+.2f}% change")
    # print(f"Overfitting Gap: {gap_reduction:+.2f}% reduction")
    


    # NEURAL NETWORK MODEL
    # Define lighter data augmentation transforms for training
    # Less aggressive augmentations suitable for medical images
    def apply_random_rotation(img):
        """Light rotation: ±5 degrees (conservative for medical images)"""
        angle = np.random.uniform(-5, 5)
        return F.rotate(img, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0)
    
    def apply_random_translation(img):
        """Light translation: up to 5% of image size (reduced from 10%)"""
        max_translate = 0.05
        tx = np.random.uniform(-max_translate, max_translate) * img.shape[-1]
        ty = np.random.uniform(-max_translate, max_translate) * img.shape[-2]
        
        # Create affine transformation matrix for translation
        theta = torch.tensor([[1, 0, tx], [0, 1, ty]], dtype=torch.float32)
        theta = theta.unsqueeze(0)
        grid = F_nn.affine_grid(theta, img.unsqueeze(0).shape, align_corners=False)
        return F_nn.grid_sample(img.unsqueeze(0), grid, align_corners=False).squeeze(0)
    
    def dynamic_augment(img):
        """Apply lighter augmentations with lower probability"""
        # 30% chance for each augmentation (reduced from 50% to be less aggressive)
        # This means most images will be lightly augmented or not augmented at all
        if np.random.random() > 0.7:  # 30% chance to rotate
            img = apply_random_rotation(img)
        if np.random.random() > 0.7:  # 30% chance to translate
            img = apply_random_translation(img)
        return img
    
    # Create datasets with lighter augmentation for training, no augmentation for validation
    train_dataset = BreastMNISTDataset(training_dataset, transform=dynamic_augment)
    val_dataset = BreastMNISTDataset(validation_dataset, transform=None)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"\nData Augmentation (Less Aggressive):")
    print(f"  Training set: Light augmentation enabled")
    print(f"    - Random rotation: ±5 degrees (30% probability)")
    print(f"    - Random translation: up to 5% (30% probability)")
    print(f"    - Each epoch will see different augmentations of the same images")
    print(f"  Validation set: No augmentation (original images)")
    
    # Train neural network
    model, train_accs, val_accs = train_neural_network(
        train_loader, val_loader, 
        num_epochs = 100, 
        learning_rate = 0.001,
        patience = 30  # Early stopping
    )
    
    # Final evaluation
    train_acc, _, _ = evaluate_neural_network(model, train_loader)
    val_acc, _, _ = evaluate_neural_network(model, val_loader)
    
    # print(f"Final Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    # print(f"Final Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    # print(f"Gap:                       {train_acc - val_acc:.4f} ({(train_acc - val_acc)*100:.2f}%)")
