import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from time import time
import matplotlib.pyplot as plt

diamonds = sns.load_dataset("diamonds")
numeric_cols = diamonds.select_dtypes(include=np.number)
diamonds_numeric = numeric_cols.copy()
def kmeans(X, k):
    """
    Perform k-means clustering on a numerical NumPy array X.
    Returns a tuple (centroids, labels).
    """
    kmeans_model = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans_model.fit(X)
    centroids = kmeans_model.cluster_centers_
    labels = kmeans_model.labels_
    return centroids, labels

def kmeans_diamonds(n, k):
    """
    Run kmeans on the first n rows of the numeric diamonds dataset.
    Returns centroids and labels.
    """
    X = diamonds_numeric.iloc[:n].to_numpy()
    return kmeans(X, k)

def kmeans_timer(n, k, n_iter=5):
    """
    Run kmeans_diamonds n_iter times and return the average runtime in seconds.
    """
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        elapsed = time() - start
        times.append(elapsed)
    return np.mean(times)

step_count = 0


def bin_search(n):
    """
    Count steps in a binary search for the worst-case element.
    Returns the index found and modifies global step_count.
    """
    global step_count
    arr = np.arange(n)
    left = 0
    right = n - 1
    x = n - 1 

    step_count = 0
    while left <= right:
        step_count += 1
        middle = left + (right - left) // 2

        if arr[middle] == x:
            return middle

        if arr[middle] < x:
            left = middle + 1
        else:
            right = middle - 1

    return -1


def plot_bin_search_steps(max_n=1024):
    """
    Plot the number of steps taken by binary search for arrays of size 1..max_n
    """
    ns = np.arange(1, max_n + 1)
    steps = []

    for n in ns:
        bin_search(n)
        steps.append(step_count)

    plt.figure(figsize=(8, 5))
    plt.plot(ns, steps, label="Binary Search Steps", color="blue")
    plt.xlabel("Array size (n)")
    plt.ylabel("Steps (worst-case)")
    plt.title("Binary Search Worst-Case Steps")
    plt.grid(True)
    plt.legend()
    plt.show()