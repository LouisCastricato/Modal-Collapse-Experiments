# This file is used to load faiss indexes and compute our variance metric

from modalcollapse.utils import get_hypersphere_points, compute_distances_from_centroid, generate_singular_value_plot
from modalcollapse.indexing.faiss_indexers import DenseFlatIndexer
import numpy as np


def default_condition(pt1, pt2):
    return True


def get_cluster_points_faiss(indexer : DenseFlatIndexer, points_per_query = 100, filter_condition = default_condition, query_points = None, return_clusters = False):
    """
    Gets points around each of the IVF centroids.
    :param index: faiss index
    :param points_per_query: number of points per query
    :param filter_condition: function that returns True if the point should be included in the output
    :param query_points: optional list of points to query, if not provided, the centroids are used
    :param return_clusters: if True, returns a list of points per cluster
    :return: numpy array of shape (points, dim) and (optional) clusterss
    """
    # gets probing points
    if query_points is None:
        query_points = indexer.centroids

    # sample the faiss index around each of these points
    indexes, _ = indexer.search_knn(query_points, points_per_query)
    output_indexes = list()
    # get the points
    pts = list()
    # lets assume for now that pts is grouped by query points, although i am not sure that this is the case lol
    for i in range(len(query_points)):
        cluster_points = list()

        for j, idx in enumerate(indexes[i]):
            # if we have multiple columns of data to return, take the 0th column as the index
            if type(idx) == tuple:
                idx = idx[0]

            # filter condition for computing our metric, helps remove noise
            if filter_condition(query_points[i],indexer.copy_of_points[idx]):
                cluster_points.append(indexer.copy_of_points[idx])

        # if this cluster was not empty, add the points back
        if len(cluster_points) > 0:
            pts.append(cluster_points)
    
    # pts is now a list of list of np.array, lets convert it a list of np.array
    if not return_clusters:
        return [np.array(pts[i]) for i in range(len(pts))]
    else:
        return [np.array(pts[i]) for i in range(len(pts))], indexes

def distance_to_centroid_faiss(indexer : DenseFlatIndexer, points_per_query = 100, filter_condition = default_condition):
    """
    Computes the variance on distance to centroid of a faiss index.
    :param index: faiss index
    :param points_per_query: number of points per query
    :return: numpy array of shape (points, dim)
    """
    # get points
    pts = get_cluster_points_faiss(indexer, points_per_query, filter_condition)
    # compute the distance to centroid for each of the points
    distances = [compute_distances_from_centroid(pts[i]) for i in range(len(pts))]
    # compute the variance of each set of distances
    variances = [np.var(distances[i]) for i in range(len(pts))]

    return np.array(variances)

def singular_value_plot_faiss(indexer : DenseFlatIndexer, points_per_query = 500, filter_condition = default_condition, return_clusters = False):
    """
    Computes an SVD diagram per cluster.
    :param index: faiss index
    :param points_per_query: number of points per query
    :param filter_condition: function that returns True if the point should be included in the output
    :param return_clusters: if True, returns a list of points per cluster
    :return: numpy array of shape (points, dim)
    """
    # get points
    pts = get_cluster_points_faiss(indexer, points_per_query, filter_condition,
        query_points=np.float32(get_hypersphere_points(set_size=256, dim=indexer.index.d)), 
        return_clusters=return_clusters)

    if return_clusters:
        pts, indexes = pts

    # compute the distance to centroid for each of the points
    singular_values = [generate_singular_value_plot(pts[i]) for i in range(len(pts))]
    if not return_clusters:
        return singular_values
    else: 
        return singular_values, indexes

def construct_faiss(dataset, data = None):
    """
    Constructs a faiss index from a numpy array
    :param dataset: numpy array of shape (n, d)
    :param data: optional data to be stored in the index, A list of length n
    :return: faiss index
    """
    indexer = DenseFlatIndexer()
    indexer.init_index(dataset.shape[1])
    # We need to associate each vector with a database id
    if data is None:
        zipped_data = list(map(lambda x: ((x[0]), x[1]), zip(range(dataset.shape[0]), list(dataset))))
    else:
        zipped_data = list(map(lambda x: ((x[0], x[1]), x[2]), zip(range(dataset.shape[0]), data, list(dataset))))

    indexer.train(dataset)
    indexer.index_data(zipped_data)
    indexer.set_copy_of_points(dataset)
    indexer.get_centroids()

    return indexer

# interpolate between the two datasets
def linear_interpolate(dataset1, dataset2):
    def interp(t):
        output = (1-t) * dataset1 + t * dataset2
        # normalize output
        output = output / np.linalg.norm(output, axis=1).reshape(-1, 1)
        return construct_faiss(output)
    return interp

def batch(datasets, data = None):
    def get_dataset(t):
        if data is None:
            return construct_faiss(datasets[t])
        else:
            return construct_faiss(datasets[t], data[t])
    return get_dataset
