# apputil.py

import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from time import time
import matplotlib.pyplot as plt

def kmeans(X, k):
    """
    Performs k-means clustering on a numerical NumPy array X.
    Returns:
        centroids (2D array): shape (k, n_features)
        labels (1D array): shape (n_samples,)
    """
    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    model.fit(X)
    centroids = model.cluster_centers_
    labels = model.labels_
    return centroids, labels


# Load the diamonds dataset from seaborn
diamonds = sns.load_dataset("diamonds")
numeric_diamonds = diamonds.select_dtypes(include=[np.number]).copy()

def kmeans_diamonds(n, k):
    """
    Runs k-means on the first n rows of the numeric diamonds dataset.
    Returns:
        centroids, labels
    """
    X = numeric_diamonds.head(n).to_numpy()
    return kmeans(X, k)

def kmeans_timer(n, k, n_iter=5):
    """
    Runs kmeans_diamonds(n, k) exactly n_iter times,
    and returns the average runtime in seconds.
    """
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        t = time() - start
        times.append(t)
    return np.mean(times)