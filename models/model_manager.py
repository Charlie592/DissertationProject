# model_manager.py
from .clustering import apply_kmeans, apply_dbscan
from .clustering import optimal_kmeans, optimal_dbscan
from sklearn.model_selection import train_test_split
from models.dimensionality_reduction import apply_optimal_pca, apply_tsne, apply_umap
from models.anomaly_detection import anomaly_detection_optimized  # Assume this function exists in your anomaly_detection.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from visualization.data_visualization import visualize_feature_importances
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def complete_analysis_pipeline(data):
    # Ensure 'data' is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data needs to be a pandas DataFrame.")
    
    # Dictionary to store reduced data for each method
    reduced_data_methods = {}
    silhouette_scores = {}

    # Apply each dimensionality reduction method
    reduced_data_methods['PCA'], _ = apply_optimal_pca(data)
    reduced_data_methods['t-SNE'] = apply_tsne(data)
    #reduced_data_methods['UMAP'] = apply_umap(data)
    
    # Calculate silhouette score for each reduced data and find the best one
    for method, reduced_data in reduced_data_methods.items():
        # Temporarily apply KMeans for silhouette score calculation; adjust based on your criteria or data
        cluster_labels = KMeans(n_clusters=5, random_state=42).fit_predict(reduced_data)
        score = silhouette_score(reduced_data, cluster_labels)
        silhouette_scores[method] = score
        print(f"{method} silhouette score: {score}")
    
    best_method = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Best dimensionality reduction method: {best_method} with a silhouette score of {silhouette_scores[best_method]}")
    
    # Perform clustering on the best reduced data
    best_reduced_data = reduced_data_methods[best_method]
    labels = choose_and_apply_clustering(best_reduced_data)
    # You can further analyze the clustering result or use it for visualization
    


def choose_and_apply_clustering(data):
    # Example simplistic criteria: Dataset size
    if len(data) > 1000:  # Assuming larger datasets might have more complex cluster shapes
        print("Using DBSCAN...")
        labels = optimal_dbscan(data)
    else:
        print("Using KMeans...")
        labels = optimal_kmeans(data)
    
    return labels




"""
    # Now you can use the feature names directly for visualization
    save_dir = '/Users/charlierobinson/Documents/Code/DissertationCode/Project 2/visualizations'
    visualize_feature_importances(tpot, X_test, y_test, feature_names, save_dir)
"""


"""
def complete_analysis_pipeline(data):
    # Ensure 'data' is a DataFrame and capture feature names before any transformation

    # Step 1: Dimensionality Reduction
    reduced_data, n_components_optimal = apply_optimal_pca(data)

    feature_names = [f'PCA_Component_{i}' for i in range(n_components_optimal)]

    # Step 2: Anomaly Detection
    anomalies_data, clean_data, predictions = anomaly_detection_optimized(reduced_data)

    # Step 3: Clustering
    labels = choose_and_apply_clustering(clean_data)
    
    # Assuming labels from clustering are used as the target variable
    X = clean_data
    y = labels

    # Splitting the dataset for supervised learning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Supervised Learning with TPOT
    tpot = TPOTClassifier(generations=2, population_size=30, verbosity=2, random_state=42, n_jobs=-1)
    tpot.fit(X_train, y_train)
    print("TPOT accuracy: ", tpot.score(X_test, y_test))
    save_dir='/Users/charlierobinson/Documents/Code/DissertationCode/Project 2/visualizations'
    visualize_feature_importances(tpot, X_test, y_test, feature_names, save_dir)"""