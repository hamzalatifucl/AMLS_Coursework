import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from medmnist import BreastMNIST
from A.augmentation import dataset_to_arrays
from A.preprocessing import preprocess_pca
from A.model import create_model, train_model
from A.evaluate import evaluate_model


def run_experiments():
    """Run all experiments"""
    training_dataset = BreastMNIST(split="train", download=True)
    validation_dataset = BreastMNIST(split="val", download=True)
    
    results = {}
    
    print("="*70, flush=True)
    print("MODEL A: RANDOM FOREST - RAW vs PROCESSED COMPARISON", flush=True)
    print("="*70, flush=True)
    
    # Experiment 1: Raw data no augmentation
    print("\n[1] Raw Data (Completely Untouched - No Preprocessing), No Augmentation", flush=True)
    X_train_raw, y_train = dataset_to_arrays(training_dataset, apply_augmentation=False)
    X_val_raw, y_val = dataset_to_arrays(validation_dataset, apply_augmentation=False)
    # Reshaped to 2D format required by random forest 
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
    X_val = X_val_raw.reshape(X_val_raw.shape[0], -1)
    model = create_model()
    model = train_model(model, X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    results['raw_no_aug'] = {'train': train_metrics, 'val': val_metrics}
    print(f"  Val - Accuracy: {val_metrics['accuracy']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}", flush=True)
    
    # Experiment 2: Preprocessed (normalization + PCA), no augmentation
    print("\n[2] Preprocessed (Normalization + PCA), No Augmentation", flush=True)
    X_train_raw, y_train = dataset_to_arrays(training_dataset, apply_augmentation=False)
    X_val_raw, y_val = dataset_to_arrays(validation_dataset, apply_augmentation=False)
    X_train_images = X_train_raw
    X_val_images = X_val_raw
    X_train, transformers = preprocess_pca(X_train_images, n_components=50, fit=True)
    X_val, _ = preprocess_pca(X_val_images, n_components=50, fit=False, pca_transformer=transformers)
    pca, scaler = transformers
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"  Features: {X_train.shape[1]} (reduced from 784 via PCA, {explained_var:.1f}% variance explained)", flush=True)
    model = create_model()
    model = train_model(model, X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    results['preprocessed_no_aug'] = {'train': train_metrics, 'val': val_metrics}
    print(f"  Val - Accuracy: {val_metrics['accuracy']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}", flush=True)
    
    # Experiment 3: Preprocessed (normalization + PCA), with augmentation
    print("\n[3] Preprocessed (Normalization + PCA), With Augmentation", flush=True)
    X_train_raw, y_train = dataset_to_arrays(training_dataset, apply_augmentation=True)
    X_val_raw, y_val = dataset_to_arrays(validation_dataset, apply_augmentation=False)
    X_train_images = X_train_raw
    X_val_images = X_val_raw
    X_train, transformers = preprocess_pca(X_train_images, n_components=50, fit=True)
    X_val, _ = preprocess_pca(X_val_images, n_components=50, fit=False, pca_transformer=transformers)
    pca, scaler = transformers
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"  Features: {X_train.shape[1]} (reduced from 784 via PCA, {explained_var:.1f}% variance explained)", flush=True)
    model = create_model()
    model = train_model(model, X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    results['preprocessed_with_aug'] = {'train': train_metrics, 'val': val_metrics}
    print(f"  Val - Accuracy: {val_metrics['accuracy']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}", flush=True)
    
    # Summary Tables
    print("\n" + "="*70, flush=True)
    print("MODEL A SUMMARY: AUGMENTATION COMPARISON (PROCESSED DATA)", flush=True)
    print("="*70, flush=True)
    print(f"{'Configuration':<20} {'Split':<8} {'Accuracy':<12} {'Recall':<12} {'F1-Score':<12}", flush=True)
    print("-" * 70, flush=True)
    for exp_name in ['processed_no_aug', 'processed_with_aug']:
        if exp_name in results:
            config_name = "No Augmentation" if 'no_aug' in exp_name else "With Augmentation"
            for split in ['train', 'val']:
                metrics = results[exp_name][split]
                print(f"{config_name:<20} {split.capitalize():<8} {metrics['accuracy']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}", flush=True)
    print("="*70, flush=True)
    
    print("\n" + "="*70, flush=True)
    print("MODEL A SUMMARY: RAW vs PROCESSED", flush=True)
    print("="*70, flush=True)
    print(f"{'Pipeline':<15} {'Split':<8} {'Accuracy':<12} {'Recall':<12} {'F1-Score':<12}", flush=True)
    print("-" * 70, flush=True)
    for exp_name in ['raw_no_aug', 'processed_no_aug']:
        if exp_name in results:
            pipeline_name = "Raw" if 'raw' in exp_name else "Processed"
            for split in ['train', 'val']:
                metrics = results[exp_name][split]
                print(f"{pipeline_name:<15} {split.capitalize():<8} {metrics['accuracy']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}", flush=True)
    print("="*70, flush=True)
    
    return results


if __name__ == "__main__":
    results = run_experiments()
