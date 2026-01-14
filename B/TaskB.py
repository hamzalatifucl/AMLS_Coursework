import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from sklearn.metrics import accuracy_score
from medmnist import BloodMNIST

# Load data
training_dataset = BloodMNIST(split="train", download=True)
validation_dataset = BloodMNIST(split="val", download=True)


class BloodMNISTDataset(Dataset):
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
            label_value = int(label[0]) if isinstance(label, np.ndarray) else int(label)
            
            self.images.append(image)
            self.labels.append(label_value)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Convert to PyTorch format 
        image = torch.FloatTensor(self.images[idx]).permute(2, 0, 1)  
        
        # Apply transform dynamically 
        if self.transform is not None:
            image = self.transform(image)
        
        label = torch.LongTensor([self.labels[idx]])[0]  
        return image, label


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(CNNClassifier, self).__init__()
        
        # layers - BloodMNIST has 3 channels (RGB), not 1 (grayscale)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
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
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
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
    
    model = CNNClassifier(num_classes=8).to(device)
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
    
    if stopped_early:
        print(f"\nStopped early at epoch {epoch+1} (saved {num_epochs - epoch - 1} epochs)")
    
    return model, train_accs, val_accs


def evaluate_neural_network(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
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
    def apply_random_rotation(img):
        """augmentation ±5 degrees"""
        angle = np.random.uniform(-5, 5)
        return F.rotate(img, angle, interpolation=F.InterpolationMode.BILINEAR, fill=0)
    
    def apply_random_translation(img):
        """augmentation up to 5% of image size"""
        max_translate = 0.05
        tx = np.random.uniform(-max_translate, max_translate) * img.shape[-1]
        ty = np.random.uniform(-max_translate, max_translate) * img.shape[-2]
        
        theta = torch.tensor([[1, 0, tx], [0, 1, ty]], dtype=torch.float32)
        theta = theta.unsqueeze(0) 
        
        # Get image shape as tuple
        img_batch = img.unsqueeze(0)  
        size = img_batch.shape
        
        # Create grid and apply transformation
        grid = F_nn.affine_grid(theta, size, align_corners=False)
        return F_nn.grid_sample(img_batch, grid, align_corners=False).squeeze(0)
    
    def dynamic_augment(img):
        if np.random.random() > 0.7:  
            img = apply_random_rotation(img)
        if np.random.random() > 0.7:
            img = apply_random_translation(img)
        return img
    
    train_dataset = BloodMNISTDataset(training_dataset, transform=dynamic_augment)
    val_dataset = BloodMNISTDataset(validation_dataset, transform=None)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # dataset information
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"Validation batches: {len(val_loader)} (batch_size={batch_size})")
    
    # Train neural network
    model, train_accs, val_accs = train_neural_network(
        train_loader, val_loader, 
        num_epochs = 100, 
        learning_rate = 0.001,
        patience = 30  # Early stopping
    )
    
    # Final evaluation on training and validation sets only
    train_acc, _, _ = evaluate_neural_network(model, train_loader)
    val_acc, _, _ = evaluate_neural_network(model, val_loader)
    
    print(f"\nFinal Evaluation Results:")
    print(f"  Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
