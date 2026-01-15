import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from medmnist import BreastMNIST
from B.dataset import BreastMNISTDataset
from B.model import CNNClassifier, DEFAULT_HYPERPARAMETERS
from B.augmentation import dynamic_augment
from B.evaluate import evaluate_model


def train_neural_network(train_loader, val_loader, hyperparameters=None, device=None, verbose=False):
    """Train CNN with early stopping returning model and training history"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if hyperparameters is None:
        hyperparameters = DEFAULT_HYPERPARAMETERS.copy()
    
    model = CNNClassifier(
        num_classes=hyperparameters['num_classes'],
        conv1_filters=hyperparameters['conv1_filters'],
        conv2_filters=hyperparameters['conv2_filters'],
        conv3_filters=hyperparameters['conv3_filters'],
        fc_size=hyperparameters['fc_size']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    weight_decay = hyperparameters.get('weight_decay', 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'], weight_decay=weight_decay)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    patience = hyperparameters['patience']
    num_epochs = hyperparameters['num_epochs']
    
    # Training history for convergence analysis
    train_losses = []
    train_accs = []
    val_accs = []
    best_epoch = 0
    
    for epoch in range(num_epochs):
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
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (best: epoch {best_epoch})")
            model.load_state_dict(best_model_state)
            break
    
    model.load_state_dict(best_model_state)
    
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_epoch': best_epoch,
        'total_epochs': len(train_accs)
    }
    
    return model, history


def run_experiments():
    """Run experiments with/without augmentation"""
    from medmnist import BreastMNIST
    
    training_dataset = BreastMNIST(split="train", download=True)
    validation_dataset = BreastMNIST(split="val", download=True)
    
    results = {}
    hyperparameters = DEFAULT_HYPERPARAMETERS.copy()
    batch_size = hyperparameters['batch_size']
    
    print("="*70, flush=True)
    print("MODEL B: CNN - AUGMENTATION COMPARISON", flush=True)
    print("="*70, flush=True)
    
    # Experiment 1: No augmentation
    print("\n[1] CNN, No Augmentation", flush=True)
    train_dataset = BreastMNISTDataset(training_dataset, transform=None)
    val_dataset = BreastMNISTDataset(validation_dataset, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model, history = train_neural_network(train_loader, val_loader, hyperparameters, verbose=True)
    print(f"  Best epoch: {history['best_epoch']}/{history['total_epochs']}", flush=True)
    train_metrics = evaluate_model(model, train_loader)
    val_metrics = evaluate_model(model, val_loader)
    results['no_aug'] = {'train': train_metrics, 'val': val_metrics, 'history': history}
    print(f"  Train - Accuracy: {train_metrics['accuracy']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}", flush=True)
    print(f"  Val   - Accuracy: {val_metrics['accuracy']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}", flush=True)
    
    # Experiment 2: With augmentation
    print("\n[2] CNN, With Augmentation", flush=True)
    train_dataset = BreastMNISTDataset(training_dataset, transform=dynamic_augment)
    val_dataset = BreastMNISTDataset(validation_dataset, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model, history = train_neural_network(train_loader, val_loader, hyperparameters, verbose=True)
    print(f"  Best epoch: {history['best_epoch']}/{history['total_epochs']}", flush=True)
    train_metrics = evaluate_model(model, train_loader)
    val_metrics = evaluate_model(model, val_loader)
    results['with_aug'] = {'train': train_metrics, 'val': val_metrics, 'history': history}
    print(f"  Train - Accuracy: {train_metrics['accuracy']:.4f}, Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}", flush=True)
    print(f"  Val   - Accuracy: {val_metrics['accuracy']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}", flush=True)
    
    # Summary Table
    print("\n" + "="*70, flush=True)
    print("MODEL B SUMMARY: AUGMENTATION COMPARISON", flush=True)
    print("="*70, flush=True)
    print(f"{'Configuration':<20} {'Split':<8} {'Accuracy':<12} {'Recall':<12} {'F1-Score':<12}", flush=True)
    print("-" * 70, flush=True)
    
    for exp_name in ['no_aug', 'with_aug']:
        if exp_name in results:
            config_name = "No Augmentation" if exp_name == 'no_aug' else "With Augmentation"
            for split in ['train', 'val']:
                metrics = results[exp_name][split]
                print(f"{config_name:<20} {split.capitalize():<8} {metrics['accuracy']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}", flush=True)
    
    print("="*70, flush=True)
    
    return results


if __name__ == "__main__":
    results = run_experiments()
    print("Model B experiments completed.")

