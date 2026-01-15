from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def evaluate_model(model, X, y, threshold=0.6):
    """Evaluate model and return metrics"""
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': preds,
        'probabilities': probs
    }

