import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from medmnist import BreastMNIST

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'A'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'B'))

from A.augmentation import dataset_to_arrays
from A.preprocessing import preprocess_pca
from A.model import create_model, train_model
from A.evaluate import evaluate_model
from B.dataset import BreastMNISTDataset
from B.model import CNNClassifier, DEFAULT_HYPERPARAMETERS as CNN_DEFAULTS
from B.augmentation import dynamic_augment
from B.evaluate import evaluate_model as evaluate_cnn
from B.train import train_neural_network


def generate_table_1():
    """Table 1: Random Forest - Feature Pipeline vs Performance"""
    training_dataset = BreastMNIST(split="train", download=True)
    validation_dataset = BreastMNIST(split="val", download=True)
    
    results = []
    
    # Raw data (completely untouched), n_estimators=100, max_depth=None
    X_train_raw, y_train = dataset_to_arrays(training_dataset, apply_augmentation=False)
    X_val_raw, y_val = dataset_to_arrays(validation_dataset, apply_augmentation=False)
    # Raw data: Pass directly with no preprocessing whatsoever
    # Only reshape to 2D format required by Random Forest (not preprocessing, just format conversion)
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_val = X_val_raw.reshape(X_val_raw.shape[0], -1)
    model = create_model({'n_estimators': 100, 'max_depth': None})
    model = train_model(model, X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    results.append({
        'pipeline': 'Flattened pixels',
        'features': 784,
        'n_estimators': 100,
        'max_depth': 'None',
        'augmentation': 'None',
        'train_acc': train_metrics['accuracy'],
        'val_acc': val_metrics['accuracy'],
        'train_precision': train_metrics['precision'],
        'val_precision': val_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'val_recall': val_metrics['recall'],
        'train_f1': train_metrics['f1'],
        'val_f1': val_metrics['f1']
    })
    
    # PCA (50 comps), n_estimators=100, max_depth=None
    # Use same raw data from above
    X_train_images = X_train_raw
    X_val_images = X_val_raw
    X_train, transformers = preprocess_pca(X_train_images, n_components=50, fit=True)
    X_val, _ = preprocess_pca(X_val_images, n_components=50, fit=False, pca_transformer=transformers)
    model = create_model({'n_estimators': 100, 'max_depth': None})
    model = train_model(model, X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    results.append({
        'pipeline': 'PCA (50 comps)',
        'features': 50,
        'n_estimators': 100,
        'max_depth': 'None',
        'augmentation': 'None',
        'train_acc': train_metrics['accuracy'],
        'val_acc': val_metrics['accuracy'],
        'train_precision': train_metrics['precision'],
        'val_precision': val_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'val_recall': val_metrics['recall'],
        'train_f1': train_metrics['f1'],
        'val_f1': val_metrics['f1']
    })
    
    # Print table
    print(f"{'Feature Pipeline':<20} {'# Features':<12} {'n_estimators':<14} {'max_depth':<12} {'Augmentation':<14} {'Train Acc':<10} {'Val Acc':<10} {'Train Prec':<10} {'Val Prec':<10} {'Train Rec':<10} {'Val Rec':<10} {'Train F1':<10} {'Val F1':<10}")
    print("-" * 100)
    for r in results:
        print(f"{r['pipeline']:<20} {r['features']:<12} {r['n_estimators']:<14} {r['max_depth']:<12} {r['augmentation']:<14} {r['train_acc']:<10.2f} {r['val_acc']:<10.2f} {r['train_precision']:<10.2f} {r['val_precision']:<10.2f} {r['train_recall']:<10.2f} {r['val_recall']:<10.2f} {r['train_f1']:<10.2f} {r['val_f1']:<10.2f}")


def generate_table_2():
    """Table 2: Random Forest - Model Capacity Analysis"""
    training_dataset = BreastMNIST(split="train", download=True)
    validation_dataset = BreastMNIST(split="val", download=True)
    
    X_train_raw, y_train = dataset_to_arrays(training_dataset, apply_augmentation=False)
    X_val_raw, y_val = dataset_to_arrays(validation_dataset, apply_augmentation=False)
    X_train_images = X_train_raw
    X_val_images = X_val_raw
    X_train, transformers = preprocess_pca(X_train_images, n_components=50, fit=True)
    X_val, _ = preprocess_pca(X_val_images, n_components=50, fit=False, pca_transformer=transformers)
    
    results = []
    
    configs = [
        {'n_estimators': 5, 'max_depth': 10},
        {'n_estimators': 10, 'max_depth': 10},
        {'n_estimators': 20, 'max_depth': 5},
        {'n_estimators': 20, 'max_depth': 10},
        {'n_estimators': 20, 'max_depth': 15},
        {'n_estimators': 30, 'max_depth': 5},
        {'n_estimators': 30, 'max_depth': 10},
        {'n_estimators': 30, 'max_depth': 15},
        {'n_estimators': 50, 'max_depth': 5},
        {'n_estimators': 50, 'max_depth': 10},
        {'n_estimators': 50, 'max_depth': 15},
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 15},
        {'n_estimators': 300, 'max_depth': None}
    ]
    
    for config in configs:
        model = create_model(config)
        model = train_model(model, X_train, y_train)
        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)
        gap = train_metrics['accuracy'] - val_metrics['accuracy']
        results.append({
            'n_estimators': config['n_estimators'],
            'max_depth': str(config['max_depth']) if config['max_depth'] else 'None',
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'gap': gap
        })
    
    # Print table
    print(f"{'n_estimators':<14} {'max_depth':<12} {'Train Acc':<12} {'Val Acc':<12} {'Overfitting Gap':<18}")
    print("-" * 100)
    for r in results:
        print(f"{r['n_estimators']:<14} {r['max_depth']:<12} {r['train_acc']:<12.2f} {r['val_acc']:<12.2f} {r['gap']:+.2f}")


def generate_table_3():
    """Table 3: Random Forest - Effect of Data Augmentation"""
    training_dataset = BreastMNIST(split="train", download=True)
    validation_dataset = BreastMNIST(split="val", download=True)
    
    results = []
    
    # No augmentation
    X_train_raw, y_train = dataset_to_arrays(training_dataset, apply_augmentation=False)
    X_val_raw, y_val = dataset_to_arrays(validation_dataset, apply_augmentation=False)
    X_train_images = X_train_raw
    X_val_images = X_val_raw
    X_train, transformers = preprocess_pca(X_train_images, n_components=50, fit=True)
    X_val, _ = preprocess_pca(X_val_images, n_components=50, fit=False, pca_transformer=transformers)
    model = create_model()
    model = train_model(model, X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    results.append({
        'augmentation': 'None',
        'train_acc': train_metrics['accuracy'],
        'val_acc': val_metrics['accuracy'],
        'train_precision': train_metrics['precision'],
        'val_precision': val_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'val_recall': val_metrics['recall'],
        'train_f1': train_metrics['f1'],
        'val_f1': val_metrics['f1']
    })
    
    # With augmentation - use SAME model as control (only variable is augmentation)
    X_train_raw, y_train = dataset_to_arrays(training_dataset, apply_augmentation=True)
    X_val_raw, y_val = dataset_to_arrays(validation_dataset, apply_augmentation=False)
    X_train_images = X_train_raw
    X_val_images = X_val_raw
    X_train, transformers = preprocess_pca(X_train_images, n_components=50, fit=True)
    X_val, _ = preprocess_pca(X_val_images, n_components=50, fit=False, pca_transformer=transformers)
    # Use same model as "no augmentation" - only variable is augmentation
    model = create_model()  # Uses default hyperparameters - same as no augmentation case
    model = train_model(model, X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    results.append({
        'augmentation': 'Rotation + Blur',
        'train_acc': train_metrics['accuracy'],
        'val_acc': val_metrics['accuracy'],
        'train_precision': train_metrics['precision'],
        'val_precision': val_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'val_recall': val_metrics['recall'],
        'train_f1': train_metrics['f1'],
        'val_f1': val_metrics['f1']
    })
    
    # Print table
    print(f"{'Augmentation':<20} {'Feature Pipeline':<20} {'Train Acc':<10} {'Val Acc':<10} {'Train Prec':<10} {'Val Prec':<10} {'Train Rec':<10} {'Val Rec':<10} {'Train F1':<10} {'Val F1':<10}")
    print("-" * 100)
    for r in results:
        print(f"{r['augmentation']:<20} {'PCA (50)':<20} {r['train_acc']:<10.2f} {r['val_acc']:<10.2f} {r['train_precision']:<10.2f} {r['val_precision']:<10.2f} {r['train_recall']:<10.2f} {r['val_recall']:<10.2f} {r['train_f1']:<10.2f} {r['val_f1']:<10.2f}")


def generate_table_4():
    """Table 4: CNN - Architecture Capacity vs Performance"""
    training_dataset = BreastMNIST(split="train", download=True)
    validation_dataset = BreastMNIST(split="val", download=True)
    
    results = []
    
    configs = [
        {'conv_layers': 2, 'filters': '16-32', 'fc_units': 64, 'conv1': 16, 'conv2': 32, 'conv3': 0, 'fc_size': 64},
        {'conv_layers': 2, 'filters': '32-64', 'fc_units': 128, 'conv1': 32, 'conv2': 64, 'conv3': 0, 'fc_size': 128},
        {'conv_layers': 3, 'filters': '32-64-128', 'fc_units': 128, 'conv1': 32, 'conv2': 64, 'conv3': 128, 'fc_size': 128}
    ]
    
    for config in configs:
        # Create custom model
        class CustomCNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if config['conv_layers'] == 2:
                    self.conv1 = torch.nn.Conv2d(1, config['conv1'], 3, padding=1)
                    self.bn1 = torch.nn.BatchNorm2d(config['conv1'])
                    self.conv2 = torch.nn.Conv2d(config['conv1'], config['conv2'], 3, padding=1)
                    self.bn2 = torch.nn.BatchNorm2d(config['conv2'])
                    self.pool = torch.nn.MaxPool2d(2, 2)
                    self.dropout_conv = torch.nn.Dropout(0.4)  # Balanced dropout
                    self.fc1 = torch.nn.Linear(config['conv2'] * 7 * 7, config['fc_size'])
                    self.fc2 = torch.nn.Linear(config['fc_size'], 2)
                    self.relu = torch.nn.ReLU()
                    self.dropout_fc = torch.nn.Dropout(0.5)  # Balanced dropout
                else:
                    self.conv1 = torch.nn.Conv2d(1, config['conv1'], 3, padding=1)
                    self.bn1 = torch.nn.BatchNorm2d(config['conv1'])
                    self.conv2 = torch.nn.Conv2d(config['conv1'], config['conv2'], 3, padding=1)
                    self.bn2 = torch.nn.BatchNorm2d(config['conv2'])
                    self.conv3 = torch.nn.Conv2d(config['conv2'], config['conv3'], 3, padding=1)
                    self.bn3 = torch.nn.BatchNorm2d(config['conv3'])
                    self.pool = torch.nn.MaxPool2d(2, 2)
                    self.dropout_conv = torch.nn.Dropout(0.4)  # Balanced dropout
                    self.fc1 = torch.nn.Linear(config['conv3'] * 3 * 3, config['fc_size'])
                    self.fc2 = torch.nn.Linear(config['fc_size'], 2)
                    self.relu = torch.nn.ReLU()
                    self.dropout_fc = torch.nn.Dropout(0.5)  # Balanced dropout
            
            def forward(self, x):
                if config['conv_layers'] == 2:
                    x = self.pool(self.relu(self.bn1(self.conv1(x))))
                    x = self.dropout_conv(x)  # Add dropout after first conv
                    x = self.pool(self.relu(self.bn2(self.conv2(x))))
                    x = self.dropout_conv(x)  # Add dropout after second conv
                    x = x.view(x.size(0), -1)
                    x = self.dropout_fc(self.relu(self.fc1(x)))
                    x = self.fc2(x)
                else:
                    x = self.pool(self.relu(self.bn1(self.conv1(x))))
                    x = self.dropout_conv(x)  # Add dropout after first conv
                    x = self.pool(self.relu(self.bn2(self.conv2(x))))
                    x = self.dropout_conv(x)  # Add dropout after second conv
                    x = self.pool(self.relu(self.bn3(self.conv3(x))))
                    x = self.dropout_conv(x)  # Add dropout after third conv
                    x = x.view(x.size(0), -1)
                    x = self.dropout_fc(self.relu(self.fc1(x)))
                    x = self.fc2(x)
                return x
        
        model = CustomCNN()
        train_dataset = BreastMNISTDataset(training_dataset, transform=None)
        val_dataset = BreastMNISTDataset(validation_dataset, transform=None)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        
        # Train model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.0005)  # Balanced - allows learning but prevents overfitting
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        patience = 12  # Balanced early stopping
        
        for epoch in range(50):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
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
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        train_metrics = evaluate_cnn(model, train_loader)
        val_metrics = evaluate_cnn(model, val_loader)
        
        results.append({
            'conv_layers': config['conv_layers'],
            'filters': config['filters'],
            'fc_units': config['fc_units'],
            'params': params,
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'train_f1': train_metrics['f1'],
            'val_f1': val_metrics['f1']
        })
    
    # Print table
    print(f"{'Conv Layers':<12} {'Filters':<15} {'FC Units':<10} {'Params (≈)':<12} {'Train Acc':<12} {'Val Acc':<12} {'Train F1':<12} {'Val F1':<12}")
    print("-" * 100)
    for r in results:
        params_k = f"{r['params']//1000}k"
        print(f"{r['conv_layers']:<12} {r['filters']:<15} {r['fc_units']:<10} {params_k:<12} {r['train_acc']:<12.2f} {r['val_acc']:<12.2f} {r['train_f1']:<12.2f} {r['val_f1']:<12.2f}")


def generate_table_5():
    """Table 5: CNN - Training Budget (Epochs)"""
    training_dataset = BreastMNIST(split="train", download=True)
    validation_dataset = BreastMNIST(split="val", download=True)
    
    train_dataset = BreastMNISTDataset(training_dataset, transform=None)
    val_dataset = BreastMNISTDataset(validation_dataset, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    results = []
    
    epochs_list = [10, 30, 45, 60, 70, 85, 100]
    
    for max_epochs in epochs_list:
        hyperparams = CNN_DEFAULTS.copy()
        hyperparams['num_epochs'] = max_epochs
        hyperparams['patience'] = max_epochs  # No early stopping for this experiment
        
        model, history = train_neural_network(train_loader, val_loader, hyperparams, verbose=False)
        
        train_metrics = evaluate_cnn(model, train_loader)
        val_metrics = evaluate_cnn(model, val_loader)
        
        results.append({
            'epochs': max_epochs,
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy']
        })
    
    # Print table
    print(f"{'Epochs':<8} {'Train Acc':<12} {'Val Acc':<12}")
    print("-" * 100)
    for r in results:
        print(f"{r['epochs']:<8} {r['train_acc']:<12.2f} {r['val_acc']:<12.2f}")


def generate_table_6():
    """Table 6: CNN - Effect of Data Augmentation"""
    training_dataset = BreastMNIST(split="train", download=True)
    validation_dataset = BreastMNIST(split="val", download=True)
    
    results = []
    
    # No augmentation
    train_dataset = BreastMNISTDataset(training_dataset, transform=None)
    val_dataset = BreastMNISTDataset(validation_dataset, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    model, history = train_neural_network(train_loader, val_loader, CNN_DEFAULTS.copy(), verbose=False)
    train_metrics = evaluate_cnn(model, train_loader)
    val_metrics = evaluate_cnn(model, val_loader)
    results.append({
        'augmentation': 'None',
        'train_acc': train_metrics['accuracy'],
        'val_acc': val_metrics['accuracy'],
        'train_precision': train_metrics['precision'],
        'val_precision': val_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'val_recall': val_metrics['recall'],
        'train_f1': train_metrics['f1'],
        'val_f1': val_metrics['f1']
    })
    
    # With augmentation
    train_dataset = BreastMNISTDataset(training_dataset, transform=dynamic_augment)
    val_dataset = BreastMNISTDataset(validation_dataset, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    model, history = train_neural_network(train_loader, val_loader, CNN_DEFAULTS.copy(), verbose=False)
    train_metrics = evaluate_cnn(model, train_loader)
    val_metrics = evaluate_cnn(model, val_loader)
    results.append({
        'augmentation': 'Rotation + Blur',
        'train_acc': train_metrics['accuracy'],
        'val_acc': val_metrics['accuracy'],
        'train_precision': train_metrics['precision'],
        'val_precision': val_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'val_recall': val_metrics['recall'],
        'train_f1': train_metrics['f1'],
        'val_f1': val_metrics['f1']
    })
    
    # Print table
    print(f"{'Augmentation':<20} {'Train Acc':<10} {'Val Acc':<10} {'Train Prec':<10} {'Val Prec':<10} {'Train Rec':<10} {'Val Rec':<10} {'Train F1':<10} {'Val F1':<10}")
    print("-" * 100)
    for r in results:
        print(f"{r['augmentation']:<20} {r['train_acc']:<10.2f} {r['val_acc']:<10.2f} {r['train_precision']:<10.2f} {r['val_precision']:<10.2f} {r['train_recall']:<10.2f} {r['val_recall']:<10.2f} {r['train_f1']:<10.2f} {r['val_f1']:<10.2f}")


def final_model():
    """Train and evaluate final models with specified configurations on test set."""
    training_dataset = BreastMNIST(split="train", download=True)
    validation_dataset = BreastMNIST(split="val", download=True)
    test_dataset = BreastMNIST(split="test", download=True)
    
    # ============================================================================
    # MODEL A: Random Forest - Final Configuration
    # ============================================================================
    # Flattened pixels, n_estimators=20, max_depth=5, with augmentation
    X_train_raw, y_train = dataset_to_arrays(training_dataset, apply_augmentation=True)
    X_val_raw, y_val = dataset_to_arrays(validation_dataset, apply_augmentation=False)
    X_test_raw, y_test = dataset_to_arrays(test_dataset, apply_augmentation=False)
    
    # Raw data: Pass directly with no preprocessing
    # Only reshape to 2D format required by Random Forest
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_val = X_val_raw.reshape(X_val_raw.shape[0], -1)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)
    
    model_a = create_model({'n_estimators': 20, 'max_depth': 5})
    model_a = train_model(model_a, X_train, y_train)
    
    train_metrics_a = evaluate_model(model_a, X_train, y_train)
    val_metrics_a = evaluate_model(model_a, X_val, y_val)
    test_metrics_a = evaluate_model(model_a, X_test, y_test)
    
    # ============================================================================
    # MODEL B: CNN - Final Configuration
    # ============================================================================
    # 3 conv layers (32-64-128 filters), 50 epochs, with augmentation
    train_dataset = BreastMNISTDataset(training_dataset, transform=dynamic_augment)
    val_dataset = BreastMNISTDataset(validation_dataset, transform=None)
    test_dataset_pytorch = BreastMNISTDataset(test_dataset, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset_pytorch, batch_size=64, shuffle=False)
    
    # Use standard CNNClassifier with 3 layers (32-64-128 filters)
    hyperparams = {
        'num_classes': 2,
        'conv1_filters': 32,
        'conv2_filters': 64,
        'conv3_filters': 128,
        'fc_size': 256,
        'learning_rate': 0.0008,
        'weight_decay': 0.0005,
        'num_epochs': 50,
        'batch_size': 64,
        'patience': 12
    }
    
    model_b, history = train_neural_network(train_loader, val_loader, hyperparams, verbose=False)
    
    train_metrics_b = evaluate_cnn(model_b, train_loader)
    val_metrics_b = evaluate_cnn(model_b, val_loader)
    test_metrics_b = evaluate_cnn(model_b, test_loader)
    
    # ============================================================================
    # Display Results
    # ============================================================================
    print("Model A: Random Forest")
    print("Configuration: Flattened pixels (784 features), n_estimators=20, max_depth=5, with augmentation")
    print(f"{'Dataset':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 100)
    print(f"{'Train':<12} {train_metrics_a['accuracy']:<12.4f} {train_metrics_a['precision']:<12.4f} {train_metrics_a['recall']:<12.4f} {train_metrics_a['f1']:<12.4f}")
    print(f"{'Validation':<12} {val_metrics_a['accuracy']:<12.4f} {val_metrics_a['precision']:<12.4f} {val_metrics_a['recall']:<12.4f} {val_metrics_a['f1']:<12.4f}")
    print(f"{'Test':<12} {test_metrics_a['accuracy']:<12.4f} {test_metrics_a['precision']:<12.4f} {test_metrics_a['recall']:<12.4f} {test_metrics_a['f1']:<12.4f}")
    print()
    print("Model B: CNN")
    print("Configuration: 3 conv layers (32-64-128 filters), 50 epochs, with augmentation")
    print(f"{'Dataset':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 100)
    print(f"{'Train':<12} {train_metrics_b['accuracy']:<12.4f} {train_metrics_b['precision']:<12.4f} {train_metrics_b['recall']:<12.4f} {train_metrics_b['f1']:<12.4f}")
    print(f"{'Validation':<12} {val_metrics_b['accuracy']:<12.4f} {val_metrics_b['precision']:<12.4f} {val_metrics_b['recall']:<12.4f} {val_metrics_b['f1']:<12.4f}")
    print(f"{'Test':<12} {test_metrics_b['accuracy']:<12.4f} {test_metrics_b['precision']:<12.4f} {test_metrics_b['recall']:<12.4f} {test_metrics_b['f1']:<12.4f}")


if __name__ == "__main__":
    final_model()
