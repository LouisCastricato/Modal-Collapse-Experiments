# This file is used to load faiss indexes and compute our variance metric

from ..utils import get_hypersphere_points, compute_distances_from_centroid
from faiss_indexers import DenseHNSWFlatIndexer
import numpy as np

def distance_to_centroid_faiss(indexer : DenseHNSWFlatIndexer, query_points = 1000, points_per_query = 100):
    """
    Computes the variance on distance to centroid of a faiss index.
    :param index: faiss index
    :param query_points: number of query points
    :param points_per_query: number of points per query
    :return: numpy array of shape (points, dim)
    """
    # get the points
    data = get_hypersphere_points(query_points, indexer.index.d)

    # sample the faiss index around each of these points
    _, indexes = indexer.search_knn(data, points_per_query)
    # get the points
    pts = list()

    # lets assume for now that pts is grouped by query points, although i am not sure that this is the case lol
    for i in range(query_points):
        pts += list()
        for idx in indexes[i]:
            pts[i].append(indexer.copy_of_points[idx])
    
    # pts is now a list of list of np.array, lets convert it a list of np.array
    pts = [np.array(pts[i]) for i in range(query_points)]

    # compute the distance to centroid for each of the points
    distances = [compute_distances_from_centroid(pts[i]) for i in range(query_points)]
    # compute the variance of each set of distances
    variances = [np.var(distances[i]) for i in range(query_points)]

    return np.array(variances)