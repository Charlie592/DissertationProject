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
from visualization.data_visualization import altair_visualize_dimensionality_reduction_and_clustering, plot_distributions_altair
import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry non-GUI backend suited for scripts and web deployment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from OpenAI import generate_summary




def complete_analysis_pipeline(data, normalized_data):
    # Ensure 'data' is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data needs to be a pandas DataFrame.")
    
    # Dictionary to store reduced data for each method
    reduced_data_methods = {}
    silhouette_scores = {}

    # Apply each dimensionality reduction method
    reduced_data_methods['PCA'], _ = apply_optimal_pca(normalized_data, 2)
    reduced_data_methods['t-SNE'] = apply_tsne(normalized_data)
    reduced_data_methods['UMAP'] = apply_umap(normalized_data)
    
    # Calculate silhouette score for each reduced data and find the best one
    for method, reduced_data in reduced_data_methods.items():
        # Temporarily apply KMeans for silhouette score calculation; adjust based on your criteria or data
        #cluster_labels = KMeans(n_clusters=5, random_state=42).fit_predict(reduced_data)
        cluster_labels = optimal_kmeans(reduced_data)
        score = silhouette_score(reduced_data, cluster_labels)
        silhouette_scores[method] = score
        #print(f"{method} silhouette score: {score}")
    
    best_method = max(silhouette_scores, key=silhouette_scores.get)
    #print(f"Best dimensionality reduction method: {best_method} with a silhouette score of {silhouette_scores[best_method]}")
    
    # Perform clustering on the best reduced data
    best_reduced_data = reduced_data_methods[best_method]
    labels = choose_and_apply_clustering(best_reduced_data)
    descriptions = generate_cluster_descriptions(data, labels)
    for description in descriptions:
        continue
        #print(description)
        #print("---")
    AI_response={}
    AI_response = generate_summary(descriptions)
    print(AI_response)

    return labels, AI_response
    


def generate_cluster_descriptions(df, cluster_labels):
    df['Cluster'] = cluster_labels
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    numeric_cols.remove('Cluster')

    all_descriptions = []

    for cluster_id in np.unique(cluster_labels):
        cluster_data = df[df['Cluster'] == cluster_id]
        other_clusters_data = df[df['Cluster'] != cluster_id]

        # Section 1: Standout Fields
        standout_desc = f"Cluster {cluster_id} standout fields are "
        standout_fields = []

        # Numeric: Top 3 based on variance
        variances = cluster_data[numeric_cols].var()
        top_numeric = variances.nlargest(3)
        for field in top_numeric.index:
            avg_val = cluster_data[field].mean()
            standout_fields.append(f"{field} (average: {avg_val:.2f})")

        # Categorical: Most significant based on frequency
        cat_diffs = {}
        for col in categorical_cols:
            mode = cluster_data[col].mode()[0]
            mode_freq = cluster_data[col].value_counts(normalize=True).get(mode, 0)
            cat_diffs[col] = mode_freq
        if cat_diffs:
            top_cat = max(cat_diffs, key=cat_diffs.get)
            top_mode = cluster_data[top_cat].mode()[0]
            standout_fields.append(f"{top_cat} (most common: {top_mode})")

        standout_desc += "; ".join(standout_fields) + "."

        # Section 2: Differentiation from Other Clusters
        diff_desc = f"\nHow Cluster {cluster_id} differs: "
        diff_fields = []
        for field in top_numeric.index:
            cluster_avg = cluster_data[field].mean()
            other_avg = other_clusters_data[field].mean()
            difference = "higher" if cluster_avg > other_avg else "lower"
            diff_fields.append(f"{field} is {difference} than the average of other clusters")

        if cat_diffs:
            mode_freq_other_clusters = other_clusters_data[top_cat].value_counts(normalize=True).get(top_mode, 0)
            freq_diff = cat_diffs[top_cat] - mode_freq_other_clusters
            freq_desc = "more common" if freq_diff > 0 else "less common"
            diff_fields.append(f"{top_mode} in {top_cat} is {freq_desc} compared to other clusters")

        diff_desc += "; ".join(diff_fields) + "."

        # Combine both sections for the cluster's description
        cluster_description = standout_desc + diff_desc
        all_descriptions.append(cluster_description)

    return all_descriptions



def choose_and_apply_clustering(data):
    # Example simplistic criteria: Dataset size
    if len(data) < 1:  # Assuming larger datasets might have more complex cluster shapes
        print("Using DBSCAN...")
        labels = optimal_dbscan(data)
    else:
        print("Using KMeans...")
        labels = optimal_kmeans(data)
    
    return labels




