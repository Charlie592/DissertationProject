#dimensionality_reduction.py

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np

def apply_optimal_pca(processed_data, variance_threshold=0.95):
    """
    Applies PCA based on a variance threshold to determine the number of components.
    """
    pca = PCA(n_components=variance_threshold)
    reduced_data = pca.fit_transform(processed_data)
    n_components_optimal = pca.n_components_
    
    print(f"Optimal number of components found: {n_components_optimal}, explaining {np.sum(pca.explained_variance_ratio_)*100:.2f}% of variance.")
    
    return reduced_data, n_components_optimal

def apply_tsne(processed_data, n_components=2, perplexity=30.0, learning_rate=200.0):
    """
    Applies t-SNE to the data, reducing it to 'n_components' dimensions.
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    reduced_data = tsne.fit_transform(processed_data)
    
    return reduced_data

def apply_umap(processed_data, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    Applies UMAP to the data, reducing it to 'n_components' dimensions.
    """
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    reduced_data = reducer.fit_transform(processed_data)
    
    return reduced_data


""" HOW TO CALL
# Assuming your_processed_data is your pre-processed dataset
pca_data, optimal_components = apply_optimal_pca(your_processed_data)
tsne_data = apply_tsne(your_processed_data)
umap_data = apply_umap(your_processed_data)"""













"""def apply_optimal_pca(processed_data, variance_threshold=0.95):
    # Determine the maximum possible number of components
    max_components = min(processed_data.shape)
    
    cumulative_variance = 0
    n_components_optimal = 0
    
    # Iteratively evaluate the number of components required to meet the variance threshold
    for n_components in range(1, max_components + 1):
        pca = PCA(n_components=n_components)
        pca.fit(processed_data)
        cumulative_variance = sum(pca.explained_variance_ratio_)
        
        # Check if the current number of components meets the variance threshold
        if cumulative_variance >= variance_threshold:
            print(f"Optimal number of components found: {n_components}, explaining {cumulative_variance*100:.2f}% of variance.")
            n_components_optimal = n_components
            break
    
    # If the loop completes without finding an optimal number, use the maximum components
    if n_components_optimal == 0:
        n_components_optimal = max(2, n_components_optimal)
        print("Warning: Applied less optimal answer.")
    
    # Apply PCA with the determined optimal number of components
    optimal_pca = PCA(n_components=n_components_optimal)
    reduced_data = optimal_pca.fit_transform(processed_data)
    
    return reduced_data, n_components_optimal
"""