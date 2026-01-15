# AMLS Assignment 2024-25

This repository contains the implementation for the Applied Machine Learning Systems (AMLS) assignment, which involves two distinct machine learning models on the BreastMNIST dataset.

## Project Overview

This project implements two classification models on the BreastMNIST dataset:
- **Model A**: Binary classification (Benign vs. Malignant) using a Random Forest classifier (Category A - Classical ML)
- **Model B**: Binary classification (Benign vs. Malignant) using a CNN (Category B - Deep Learning)

## Project Structure

```
AMLS_Coursework/
├── A/                      # Model A (Random Forest) implementation
│   ├── model.py            # Random Forest model definition and hyperparameters
│   ├── preprocessing.py    # Preprocessing pipelines (raw vs PCA)
│   ├── augmentation.py     # Data augmentation functions (rotation, blur)
│   ├── evaluate.py         # Evaluation metrics (accuracy, precision, recall, F1)
│   ├── train.py            # Training and experimentation script
│   └── test.py             # Test set evaluation script (dormant)
├── B/                      # Model B (CNN) implementation
│   ├── model.py            # CNN architecture and hyperparameters
│   ├── dataset.py           # PyTorch dataset class for BreastMNIST
│   ├── augmentation.py     # Data augmentation functions (rotation, blur)
│   ├── evaluate.py         # Evaluation metrics (accuracy, precision, recall, F1)
│   ├── train.py            # Training and experimentation script
│   └── test.py             # Test set evaluation script (dormant)
├── Datasets/               # Dataset folder (empty for submission)
│   └── BreastMNIST/        # Will contain BreastMNIST dataset during assessment
├── main.py                 # Main script - runs final models and evaluates on test set
├── requirements.txt        # Python package dependencies
└── README.md               # This file
```

## File Descriptions

### `main.py`
The main entry point for the project. Running this script will:
- Train Model A (Random Forest) with final configuration: flattened pixels, n_estimators=20, max_depth=5, with augmentation
- Train Model B (CNN) with final configuration: 3 conv layers (32-64-128 filters), 50 epochs, with augmentation

### `A/` - Model A (Random Forest)
- **`model.py`**: Random Forest model definition with default and augmentation-specific hyperparameters
- **`preprocessing.py`**: Preprocessing pipelines (raw untouched images vs PCA with normalization)
- **`augmentation.py`**: Data augmentation functions (rotation ±3°, blur with σ=0.3)
- **`evaluate.py`**: Evaluation functions returning accuracy, precision, recall, F1-score with adjustable decision threshold
- **`train.py`**: Training script that runs experiments comparing raw vs processed, and with/without augmentation

### `B/` - Model B (CNN)
- **`model.py`**: CNN architecture (3 conv layers: 32-64-128 filters) with hyperparameters
- **`dataset.py`**: PyTorch dataset class for BreastMNIST with optional augmentation
- **`augmentation.py`**: Data augmentation functions (rotation ±3°, blur with σ=0.3) for PyTorch tensors
- **`evaluate.py`**: Evaluation functions returning accuracy, precision, recall, F1-score
- **`train.py`**: Training script with early stopping and best model saving

### `Datasets/`
This folder is empty for submission


### Data Augmentation
- **Applied To**: Training set only (validation remains unaugmented)
- **Techniques**:
  - Random rotation: ±3 degrees 
  - Blur with σ=0.3 
- **Model A**: Augmentation doubles the training set size (each original image + one augmented copy)
- **Model B**: Augmentation applied dynamically during training (different augmentations each epoch)

## Usage

### Running Final Models

To train and evaluate the final models on all datasets:

```bash
python main.py
```

This will:
- Train Model A (Random Forest) with final configuration
- Train Model B (CNN) with final configuration  
- Evaluate both models on train, validation, and test sets
- Display formatted results table


## Final Model Configurations

### Model A: Random Forest
- **Features**: Flattened pixels (784 features - raw, no preprocessing)
- **Hyperparameters**: 
  - `n_estimators=20`
  - `max_depth=5`
  - `min_samples_split=40`
  - `min_samples_leaf=20`
  - `max_features='sqrt'`
- **Augmentation**: Yes (rotation + blur)
- **Decision Threshold**: 0.6 (to reduce false positives)

### Model B: CNN
- **Architecture**: 3 convolutional layers (32 → 64 → 128 filters)
- **Fully Connected**: 256 → 128 → 2 units
- **Hyperparameters**:
  - Learning rate: 0.0008
  - Weight decay: 0.0005
  - Batch size: 64
  - Epochs: 50 (with early stopping, patience=12)
- **Regularization**: Dropout (0.4 after conv, 0.5 after FC), L2 weight decay, batch normalization
- **Augmentation**: Yes (rotation + blur, applied dynamically during training)


## Required Packages

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Key packages:
- `torch` - PyTorch for CNN implementation
- `torchvision` - Additional PyTorch utilities
- `medmnist` - BreastMNIST dataset loader
- `scikit-learn` - Random Forest classifier and preprocessing (PCA, StandardScaler)
- `numpy` - Array operations
- `scipy` - Image augmentation (rotation, blur)
