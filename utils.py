# implements a number of helper functions to compute k-means clusters and related

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
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

def compute_distances_from_centroid(data):
    """
    Computes the distances of each data point from the centroid of the data.
    :param data: numpy array of shape (n, d)
    :return: numpy array of shape (n, 1)
    """
    # compute centroid
    centroid = np.mean(data, axis=0)
    # compute distances. Cosine similarity is not very good, back to L2
    return 1 - np.dot(data, centroid)

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

def get_normalized_hypercube_points(set_size=1000, dim=512):
    """
    Generates a set of points on a hypercube of dimension dim.
    """
    output = np.array([np.random.uniform(-1, 1, dim) for _ in range(set_size)])
    output = output / np.linalg.norm(output, axis=1).reshape(-1, 1)
    return output

def get_splooch_points(set_size=1000, dim=512, splooches=10, scale_upper_bound=1.0):
    """
    Generates a set of points that splooch the surface of a hyper sphere
    """
    points_per_splooch = set_size // splooches
    splooch_list = list()
    for _ in range(splooches):
        splooch = get_hypersphere_points(points_per_splooch, dim)
        random_direction = np.random.uniform(-1, 1, dim)
        random_scale = np.random.uniform(0, scale_upper_bound)

        splooch = splooch * random_scale + random_direction

        # normalize splooch
        splooch = splooch / np.linalg.norm(splooch, axis=1).reshape(-1, 1)
        # append
        splooch_list.append(splooch)

    # concatenate list of numpy arrays into numpy array
    return np.concatenate(splooch_list)


def generate_a_random_rotation_matrix(dim=512):
    """
    Generates a random rotation matrix of dimension dim.
    """
    # uses the orthogonal group of dimension dim
    return ortho_group.rvs(dim)


def generate_singular_value_plot(data, k=2, size=10000):
    """
    Generates a singular value plot for the given data.
    :param data: numpy array of shape (n, d)
    :param k: number of singular values to plot
    :param size: number of points to use for the plot
    """
    # if len(data) > size, sample data
    if len(data) > size:
        idxs = np.random.choice(range(len(data)), size, replace=False)
        data = data[idxs]
    # compute SVD
    _, s, _ = np.linalg.svd(data)
    # plot
    if k is None:
        return np.log(s)
    else:
        return np.log(s[:k])
