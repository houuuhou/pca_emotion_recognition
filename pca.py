import numpy as np
import matplotlib.pyplot as plt
import os

def plot_correlation_circle(loadings, feature_names, explained_variance):
    """Plot the correlation circle for PCA"""
    # Visualization 2: Correlation Circle (Fixed)
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax.add_artist(circle)
    
    # Plot variable vectors (using loadings)
    for i in range(len(feature_names)):
        x, y = loadings[i, 0], loadings[i, 1]
        
        # Ensure vectors reach circle boundary (normalize if needed)
        norm = np.sqrt(x**2 + y**2)
        if norm > 1.0:  # Shouldn't happen with proper loadings
            x, y = x/norm, y/norm
        
        ax.arrow(0, 0, x, y, head_width=0.05, color='red', alpha=0.7)
        ax.text(x*1.15, y*1.15, feature_names[i], color='blue', 
                ha='center', va='center')
    
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
    plt.title("Correlation Circle (Variable Loadings)")
    plt.grid()
    
    # Verification
    print("\nVariable Loadings (Correlations with PCs):")
    for i, name in enumerate(feature_names):
        norm = np.linalg.norm(loadings[i, :2])
        print(f"{name}: Norm={norm:.4f}, PC1={loadings[i,0]:.4f}, PC2={loadings[i,1]:.4f}")
    
    plt.show()

def manual_pca(features, feature_names=None, labels=None, save_path="pca_artifacts"):
    """Perform PCA and save transformation parameters"""
    # Standardization
    means = np.mean(features, axis=0)
    X_centered = features - means
    stds = np.std(X_centered, axis=0, ddof=0)
    X_std = X_centered / stds

    # PCA
    m = X_std.shape[0]
    R = (X_std.T @ X_std) / m
    eigvals, eigvecs = np.linalg.eig(R)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    scores = X_std @ eigvecs
    explained_variance = eigvals / np.sum(eigvals)
    loadings = eigvecs * np.sqrt(eigvals.reshape(1, -1))

    # Save artifacts
    os.makedirs(save_path, exist_ok=True)
    np.save(f"{save_path}/pca_mean.npy", means)
    np.save(f"{save_path}/pca_std.npy", stds)
    np.save(f"{save_path}/pca_components.npy", eigvecs)
    np.save(f"{save_path}/pca_scores.npy", scores)
    
    # Plot correlation circle if feature names are provided
    if feature_names is not None:
        plot_correlation_circle(loadings[:, :2], feature_names, explained_variance)
    
    return {
        'scores': scores,
        'loadings': loadings,
        'explained_variance': explained_variance,
        'mean': means,
        'std': stds
    }

def apply_pca(features, pca_path="pca_artifacts"):
    """Apply PCA transformation to new data"""
    means = np.load(f"{pca_path}/pca_mean.npy")
    stds = np.load(f"{pca_path}/pca_std.npy")
    components = np.load(f"{pca_path}/pca_components.npy")
    
    X_centered = features - means
    X_std = X_centered / stds
    return X_std @ components