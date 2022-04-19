from utils import *
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool

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
    data_points = 10000
    dim = 128
    cluster_count = 100
    rotation_count = 1

    # generate data
    hypersphere_data = generate_data(data_points, dim, rotation_count,
        generate_function=get_hypersphere_points)
    splooch_data = generate_data(data_points, dim, rotation_count,
        generate_function=partial(get_splooch_points, scale_upper_bound=0.25, splooches=100))

    # interpolate between the two datasets
    def linear_interpolate(dataset1, dataset2):
        def interp(t):
            output = dataset1 * (1 - t) + dataset2 * t
            # normalize output
            output = output / np.linalg.norm(output, axis=1).reshape(-1, 1)
            return output

        return interp

    # interpolate
    interp = linear_interpolate(hypersphere_data, splooch_data)
    # multiprocessing
    pool = Pool(processes=8)
    # get variance
    variances = pool.map(partial(get_k_means_variance, k=cluster_count), [interp(i/10) for i in range(10)])
    pool.close()


    
    for idx, var in enumerate(variances):
        # compute a histogram using matplotlib
        plt.hist(var)
        plt.show()
        plt.savefig('graphs/variance_interp_' + str(idx) + '.png')
        plt.clf()

