"""
Clustering and dimensionality reduction utilities for embedding analysis.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, MiniBatchKMeans


def cluster_embeddings(embeddings, target_samples):
    """
    Reduce number of samples using K-means clustering.
    
    Args:
        embeddings: Input embeddings (torch.Tensor or np.ndarray)
        target_samples: Target number of samples
        
    Returns:
        Cluster centers (np.ndarray)
    """
    num_samples = embeddings.shape[0]

    if num_samples <= target_samples:
        if isinstance(embeddings, torch.Tensor):
            return embeddings.cpu().detach().numpy()
        return embeddings

    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()

    kmeans = MiniBatchKMeans(
        n_clusters=target_samples, 
        random_state=42, 
        batch_size=256, 
        n_init=1, 
        max_iter=50
    )
    kmeans.fit(embeddings)

    return kmeans.cluster_centers_


def clustering(X, pca=False, n_clusters=10, n_components=10):
    """
    Apply PCA (optional) and K-means clustering to embeddings.
    
    Args:
        X: Input data (torch.Tensor or np.ndarray)
        pca: Whether to apply PCA before clustering
        n_clusters: Number of clusters
        n_components: Number of PCA components
        
    Returns:
        labels: Cluster assignments
        X_processed: Processed data (after PCA if applied)
    """
    # Convert to numpy if needed
    if not isinstance(X, np.ndarray):
        X = X.cpu().detach().numpy()
    
    X = np.nan_to_num(X)
    
    # Flatten if needed
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    
    # Apply PCA if requested
    if pca:
        X = normalize(X)
        # n_components cannot exceed the number of features or samples
        n_components = min(n_components, X.shape[1], X.shape[0])
        if n_components > 0:
            X = PCA(n_components=n_components).fit_transform(X)

    # n_clusters cannot exceed number of samples
    n_clusters = min(n_clusters, X.shape[0])

    kmeans = KMeans(
        n_clusters=n_clusters, 
        n_init=1, 
        max_iter=50, 
        random_state=42
    ).fit(X)
    
    return kmeans.labels_, X
