# clustering.py

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import numpy as np
import matplotlib.pyplot as plt


def optimal_kmeans(data):
    wcss = []
    for i in range(1, 11):
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    # Automatically find the elbow point
    kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
    n_clusters_optimal = kn.knee
    print(f"Optimal number of clusters: {n_clusters_optimal}")
    
    return apply_kmeans(data, n_clusters=n_clusters_optimal)


def apply_kmeans(data, n_clusters=3):
    # Perform K-means clustering with specified number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

def optimal_dbscan(data):
    # Find the nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    
    distances = np.sort(distances, axis=0)[:,1]
    
    # Automatically find the knee point as the optimal eps value
    kn = KneeLocator(np.arange(len(distances)), distances, curve='convex', direction='increasing')
    eps_optimal = distances[kn.knee]
    print(f"Optimal eps value: {eps_optimal}")
    
    return apply_dbscan(data, eps=eps_optimal, min_samples=5)

def apply_dbscan(data, eps=0.5, min_samples=5):
    # Perform DBSCAN clustering with specified eps and min_samples values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels


