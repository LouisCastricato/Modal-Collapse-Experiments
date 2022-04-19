from utils import *
import matplotlib.pyplot as plt
from functools import partial
from indexing.faiss_utils import distance_to_centroid_faiss
from indexing.faiss_indexers import DenseFlatIndexer
from tqdm import tqdm
from scipy.stats import skew, kurtosis

def generate_data(data_points=1000, dim=512, rotation_count=1, generate_function=get_hypersphere_points):
    # get hyper sphere points
    data = generate_function(data_points, dim)

    # get a random rotation matrix
    R = [generate_a_random_rotation_matrix(dim) for _ in range(rotation_count)]
    
    # apply the rotation matrix to the data
    rotated_data = [data]
    for _ in range(rotation_count):
        choice = np.random.choice(len(R), 1)[0]
        rotated_data.append(np.dot(rotated_data[-1], R[choice]))

    # concat
    data = np.reshape(np.stack(rotated_data, axis=0), (len(rotated_data) * data_points, -1))
    # random sample
    return data[np.random.choice(data_points*(rotation_count+1), data_points, replace=False)]

def get_k_means_variance(data, k):
    """
    Computes the variance of the k-means clustering on the given data.
    :param data: numpy array of shape (n, d)
    :param k: number of clusters
    """
    clusters = compute_kmeans(data, k=k)

    def get_cluster(cluster_idx):
        return get_where_index(data, clusters, cluster_idx)

    # get the variance of each cluster
    return [np.var(compute_distances_from_centroid(get_cluster(i))) for i in range(k)]


if __name__ == '__main__':

    # constants
    data_points = 1000000
    dim = 128
    cluster_count = 100
    rotation_count = 1

    # generate data
    hypersphere_data = np.float32(generate_data(data_points, dim, rotation_count,
        generate_function=get_hypersphere_points))
    splooch_data = np.float32(generate_data(data_points, dim, rotation_count,
        generate_function=partial(get_splooch_points, scale_upper_bound=0.25, splooches=100)))

    # interpolate between the two datasets
    def linear_interpolate(dataset1, dataset2):

        def interp(t):
            output = dataset1 * (1 - t) + dataset2 * t
            # normalize output
            output = output / np.linalg.norm(output, axis=1).reshape(-1, 1)
            return output

        def construct_faiss(interp):
            # create a faiss index from linear interpolation data data
            indexer = DenseFlatIndexer()
            indexer.init_index(dim)
            # We need to associate each vector with a database id
            zipped_data = list(map(lambda x: x, zip(range(interp.shape[0]), list(interp))))
            indexer.index_data(zipped_data)
            indexer.get_copy_of_points()
            return indexer

        def compose(t):
            return construct_faiss(interp(t))

        return compose

    # interpolate
    interp = linear_interpolate(hypersphere_data, splooch_data)

    # get variance
    variances = list()
    for i in tqdm(range(10)):
        variances.append(distance_to_centroid_faiss(interp(float(i)/10.)))

    for idx, var in enumerate(variances):
        # compute a histogram using matplotlib
        print("Skew, kurtosis:", skew(var), kurtosis(var))
        plt.hist(var)
        plt.show()
        plt.savefig('graphs/variance_interp_' + str(idx) + '.png')
        plt.clf()

