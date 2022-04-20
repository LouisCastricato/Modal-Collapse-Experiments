# This file is used to load faiss indexes and compute our variance metric

from utils import get_hypersphere_points, compute_distances_from_centroid
from indexing.faiss_indexers import DenseFlatIndexer
import numpy as np


def default_condition(pt1, pt2):
    return True

def distance_to_centroid_faiss(indexer : DenseFlatIndexer, points_per_query = 100, filter_condition = default_condition):
    """
    Computes the variance on distance to centroid of a faiss index.
    :param index: faiss index
    :param points_per_query: number of points per query
    :return: numpy array of shape (points, dim)
    """
    # gets probing points
    query_points = indexer.centroids

    # sample the faiss index around each of these points
    indexes, _ = indexer.search_knn(query_points, points_per_query)

    # get the points
    pts = list()
    # lets assume for now that pts is grouped by query points, although i am not sure that this is the case lol
    for i in range(len(query_points)):
        cluster_points = list()

        for idx in indexes[i]:
            if filter_condition(query_points[i],indexer.copy_of_points[idx]):
                cluster_points.append(indexer.copy_of_points[idx])
        if len(cluster_points) > 0:
            pts.append(cluster_points)
    
    # pts is now a list of list of np.array, lets convert it a list of np.array
    pts = [np.array(pts[i]) for i in range(len(pts))]

    # compute the distance to centroid for each of the points
    distances = [compute_distances_from_centroid(pts[i]) for i in range(len(pts))]
    # compute the variance of each set of distances
    variances = [np.mean(distances[i]) for i in range(len(pts))]

    return np.array(variances)

def construct_faiss(dataset):
    """
    Constructs a faiss index from a numpy array
    :param dataset: numpy array of shape (n, d)
    :return: faiss index
    """
    indexer = DenseFlatIndexer()
    indexer.init_index(dataset.shape[1])
    # We need to associate each vector with a database id
    zipped_data = list(map(lambda x: x, zip(range(dataset.shape[0]), list(dataset))))
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

def batch(datasets):
    def get_dataset(t):
        return construct_faiss(datasets[t])
    return get_dataset