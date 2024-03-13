# model_manager.py
from .clustering import apply_kmeans, apply_dbscan
from .clustering import optimal_kmeans, optimal_dbscan
from sklearn.model_selection import train_test_split
from models.dimensionality_reduction import apply_optimal_pca, apply_tsne, apply_umap
from models.anomaly_detection import anomaly_detection_optimized  # Assume this function exists in your anomaly_detection.py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from visualization.data_visualization import altair_visualize_dimensionality_reduction_and_clustering, visualize_anomalies_with_predictions


def complete_analysis_pipeline(data):
    # Ensure 'data' is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data needs to be a pandas DataFrame.")
    
    # Dictionary to store reduced data for each method
    reduced_data_methods = {}
    silhouette_scores = {}

    # Apply each dimensionality reduction method
    reduced_data_methods['PCA'], _ = apply_optimal_pca(data, 2)
    reduced_data_methods['t-SNE'] = apply_tsne(data)
    #reduced_data_methods['UMAP'] = apply_umap(data)
    
    # Calculate silhouette score for each reduced data and find the best one
    for method, reduced_data in reduced_data_methods.items():
        # Temporarily apply KMeans for silhouette score calculation; adjust based on your criteria or data
        #cluster_labels = KMeans(n_clusters=5, random_state=42).fit_predict(reduced_data)
        cluster_labels = optimal_kmeans(reduced_data)
        score = silhouette_score(reduced_data, cluster_labels)
        silhouette_scores[method] = score
        print(f"{method} silhouette score: {score}")
    
    best_method = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Best dimensionality reduction method: {best_method} with a silhouette score of {silhouette_scores[best_method]}")
    
    # Perform clustering on the best reduced data
    best_reduced_data = reduced_data_methods[best_method]
    labels = choose_and_apply_clustering(best_reduced_data)
    # You can further analyze the clustering result or use it for visualization

    method_name = {best_method}  # or 't-SNE', depending on which method was used
    feature_names = ['Principal Component 1', 'Principal Component']  # Adjust based on your actual feature names
    print(reduced_data.shape)
    chart = altair_visualize_dimensionality_reduction_and_clustering(reduced_data, labels, method_name, feature_names)
    chart.save('dimensionality_reduction_clustering_visualization.html')
    anomalies_data, normal_data, predictions = anomaly_detection_optimized(reduced_data)
    visualize_anomalies_with_predictions(reduced_data, predictions)
    

def choose_and_apply_clustering(data):
    # Example simplistic criteria: Dataset size
    if len(data) < 1000:  # Assuming larger datasets might have more complex cluster shapes
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