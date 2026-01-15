import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def preprocess_raw(images):
    """Raw data"""
    X_flattened = images.reshape(images.shape[0], -1)  # Flatten to 784 features
    return X_flattened, None  # No transformer needed for raw


def preprocess_pca(images, n_components=50, fit=True, pca_transformer=None):
    """Flattening, normalization and feature extraction
    """
    X_flattened = images.reshape(images.shape[0], -1)  # Flatten to 784 features
    
    # Normalization 
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flattened)
        # Feature extraction dimensionality reduction
        pca = PCA(n_components=n_components, random_state=0)
        X_pca = pca.fit_transform(X_scaled)
        return X_pca, (pca, scaler)
    else:
        if pca_transformer is None:
            raise ValueError("pca_transformer required when fit=False")
        pca, scaler = pca_transformer
        X_scaled = scaler.transform(X_flattened)
        X_pca = pca.transform(X_scaled)
        return X_pca, (pca, scaler)

