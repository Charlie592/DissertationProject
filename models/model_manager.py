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
from visualization.data_visualization import plot_distributions_altair
import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry non-GUI backend suited for scripts and web deployment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from OpenAI import generate_summary
import streamlit as st


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
        # Temporarily apply KMeans for silhouette score calculation; adjust based on your 
        #criteria or data
        #trend_labels = KMeans(n_trends=5, random_state=42).fit_predict(reduced_data)
        trend_labels = optimal_kmeans(reduced_data)
        score = silhouette_score(reduced_data, trend_labels)
        silhouette_scores[method] = score
        #print(f"{method} silhouette score: {score}")
    
    best_method = max(silhouette_scores, key=silhouette_scores.get)
    #print(f"Best dimensionality reduction method: {best_method} with a 
    #silhouette score of {silhouette_scores[best_method]}")
    
    # Perform trending on the best reduced data
    best_reduced_data = reduced_data_methods[best_method]
    labels = choose_and_apply_trending(best_reduced_data)
    descriptions = generate_trend_descriptions(data, labels)
    AI_response={}
    AI_response = generate_summary(descriptions)
    print(AI_response)

    return labels, AI_response
    


def generate_trend_descriptions(data, trend_labels, numeric_metric='var', diff_metric='mean'):
    data['trend'] = trend_labels
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=np.number).columns.tolist()
    numeric_cols.remove('trend')

    all_descriptions = []

    for trend_id in np.unique(trend_labels):
        trend_data = data[data['trend'] == trend_id]
        other_trends_data = data[data['trend'] != trend_id]

        # Section 1: Standout Fields
        standout_desc = f"trend {trend_id+1} standout fields are "
        standout_fields = []

        # Numeric: Based on specified metric
        if numeric_metric == 'var':
            metric_values = trend_data[numeric_cols].var()
        elif numeric_metric == 'std':
            metric_values = trend_data[numeric_cols].std()
        else:
            raise ValueError("Unsupported numeric_metric provided.")
        
        top_numeric = metric_values.nlargest(3)
        for field in top_numeric.index:
            avg_val = trend_data[field].mean()
            standout_fields.append(f"{field} (average: {avg_val:.2f})")

        # Categorical: Most significant based on frequency
        if categorical_cols:
            cat_diffs = {}
            for col in categorical_cols:
                mode = trend_data[col].mode()[0] if not trend_data[col].mode().empty else 'N/A'
                mode_freq = trend_data[col].value_counts(normalize=True).get(mode, 0)
                cat_diffs[col] = mode_freq
            if cat_diffs:
                top_cat = max(cat_diffs, key=cat_diffs.get)
                top_mode = trend_data[top_cat].mode()[0] if not trend_data[top_cat].mode().empty else 'N/A'
                standout_fields.append(f"{top_cat} (most common: {top_mode})")

        standout_desc += "; ".join(standout_fields) + "."

        # Section 2: Differentiation from Other trends
        diff_desc = f"\nHow trend {trend_id+1} differs: "
        diff_fields = []
        for field in top_numeric.index:
            trend_avg = trend_data[field].mean() if diff_metric == 'mean' else trend_data[field].median()
            other_avg = other_trends_data[field].mean() if diff_metric == 'mean' else other_trends_data[field].median()
            difference = "higher" if trend_avg > other_avg else "lower"
            diff_val = abs(trend_avg - other_avg)
            diff_fields.append(f"{field} is {difference} than the average of other trends by {diff_val:.2f}")

        if categorical_cols and cat_diffs:
            mode_freq_other_trends = other_trends_data[top_cat].value_counts(normalize=True).get(top_mode, 0)
            freq_diff = cat_diffs[top_cat] - mode_freq_other_trends
            freq_desc = "more common" if freq_diff > 0 else "less common"
            diff_fields.append(f"{top_mode} in {top_cat} is {freq_desc} compared to other trends")

        diff_desc += "; ".join(diff_fields) + "."

        # Combine both sections for the trend's description
        trend_description = standout_desc + diff_desc
        all_descriptions.append(trend_description)

    return all_descriptions


def choose_and_apply_trending(data):
    # Example simplistic criteria: Dataset size
    if len(data) < 1:  # Assuming larger datasets might have more complex trend shapes
        print("Using DBSCAN...")
        labels = optimal_dbscan(data)
    else:
        print("Using KMeans...")
        labels = optimal_kmeans(data)
    
    return labels




