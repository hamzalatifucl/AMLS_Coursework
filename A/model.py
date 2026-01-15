from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


DEFAULT_HYPERPARAMETERS = {
    'n_estimators': 200,
    'max_depth': 8,               # Reduced depth to prevent overfitting
    'min_samples_split': 50,      # Increased from 40 for better regularization with augmentation
    'min_samples_leaf': 25,       # Increased from 20 for better regularization with augmentation
    'max_features': 'sqrt',
    'random_state': 0
}


def create_model(hyperparameters=None):
    """Create Random Forest model with hyperparameters."""
    if hyperparameters is None:
        hyperparameters = DEFAULT_HYPERPARAMETERS.copy()
    
    return RandomForestClassifier(
        n_estimators=hyperparameters.get('n_estimators', DEFAULT_HYPERPARAMETERS['n_estimators']),
        max_depth=hyperparameters.get('max_depth', DEFAULT_HYPERPARAMETERS['max_depth']),
        min_samples_split=hyperparameters.get('min_samples_split', DEFAULT_HYPERPARAMETERS['min_samples_split']),
        min_samples_leaf=hyperparameters.get('min_samples_leaf', DEFAULT_HYPERPARAMETERS['min_samples_leaf']),
        max_features=hyperparameters.get('max_features', DEFAULT_HYPERPARAMETERS['max_features']),
        random_state=hyperparameters.get('random_state', DEFAULT_HYPERPARAMETERS['random_state']),
        n_jobs=-1
    )


def train_model(model, X_train, y_train):
    """Training Random Forest model"""
    model.fit(X_train, y_train)
    return model

