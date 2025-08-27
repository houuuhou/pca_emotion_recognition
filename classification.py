import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from feature_extraction import extract_dataset_features
from pca import manual_pca,plot_correlation_circle

def visualize_pca_results(scores, labels, explained_variance):
    """Show PCA visualization plots"""
    # 1. Scatter plot of PC1 vs PC2
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(scores[mask, 0], scores[mask, 1], color=color,
                   label=label, alpha=0.7)
    
    plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")
    plt.title("PCA Projection (PC1 vs PC2)")
    plt.legend()
    plt.grid()
    plt.show()
    # Create integer x-axis values
    n_components = len(explained_variance)
    components = np.arange(1, n_components + 1, dtype=int)

    
    # 2. Enhanced Scree Plot with elbow detection
    plt.figure(figsize=(10, 5))
    
    # Variance expliquée individuelle
    plt.plot(components, explained_variance, 'bo-')
    plt.title('Graphique des Eboulis')
    plt.xlabel('Composantes Principales')
    plt.ylabel('Variance Expliquée')
    plt.xticks(components) 
    plt.grid()
    plt.show()
    
    # Variance cumulative
    plt.figure(figsize=(10, 5))
    cumulative = np.cumsum(explained_variance)
    plt.plot(components, cumulative, 'ro-')
    plt.title('Variance Cumulative')
    plt.xlabel('Nombre de Composantes')
    plt.ylabel('Variance Expliquée Cumulative')
    plt.axhline(y=0.8, color='g', linestyle='--', label='80% Variance')
    plt.xticks(components) 
    plt.grid()
    plt.legend()
    plt.show()


def train_model(dataset_path="dataset"):
    # 1. Extract features
    features, labels, feature_names= extract_dataset_features(dataset_path)
    
    # 2. Perform PCA
    pca_results = manual_pca(features)
    scores = pca_results['scores'][:, :5]  # Use first 4 components
    num_features = features.shape[1]
    # 3. Visualize PCA results
    visualize_pca_results(scores, labels, pca_results['explained_variance'])
    plot_correlation_circle(pca_results['loadings'], feature_names, pca_results['explained_variance'])
    
    # 4. Train classifier
    X_train, X_test, y_train, y_test = train_test_split(
        scores, labels, test_size=0.2, random_state=90)
    
    svm = SVC(kernel='rbf', C=10, gamma=0.1, random_state=30)
    svm.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = svm.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(labels),
                yticklabels=np.unique(labels))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # 6. Save model
    joblib.dump(svm, 'expression_classifier.joblib')
    print("Model trained and saved successfully")

if __name__ == "__main__":
    train_model()