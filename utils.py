# implements a number of helper functions to compute k-means clusters and related

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import ortho_group
import skdim.datasets

def compute_kmeans(data, k):
    """
    Computes k-means clustering on the given data.
    :param data: numpy array of shape (n, d)
    :param k: number of clusters
    :return: numpy array of shape (n, k)
    """
    kmeans = KMeans(n_clusters=k).fit(data)
    return kmeans

def get_where_index(data, km, label_index):
    """
    Returns the index of the data points that belong to the given cluster.
    :param data: numpy array of shape (n, d)
    :param km: k-means object
    :param label_index: int
    """
    return data[np.where(km.labels_ == label_index)]

def get_hypersphere_points(set_size=1000, dim=512):
    """
    Generates a set of points on a hypersphere of dimension dim.
    """
    return skdim.datasets.hyperSphere(set_size, dim)

def generate_a_random_rotation_matrix(dim=512):
    """
    Generates a random rotation matrix of dimension dim.
    """
    # uses the orthogonal group of dimension dim
    return ortho_group.rvs(dim)

def compute_sample_skew_CI(n):
    """
    Computes the confidence interval for the sample skew of the given data.
    :param n: number of samples
    :return: numpy array of shape (2, 1)
    """
    return np.array([np.sqrt(n) / np.sqrt(n - 1), np.sqrt(n) / np.sqrt(n - 1)])

def flatten(t):
    """
    Flattens a list of lists.
    """
    return [item for sublist in t for item in sublist]