from utils import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # constants
    data_points = 10000
    dim = 512
    cluster_count = 100
    rotation_count = 100

    # get hyper sphere points
    data = get_hypersphere_points(data_points, dim)

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
    data = data[np.random.choice(data_points*(rotation_count+1), data_points, replace=False)]

    clusters = compute_kmeans(data, k=cluster_count)

    def get_cluster(cluster_idx):
        return get_where_index(data, clusters, cluster_idx)

    # get the variance of each cluster
    var = [np.var(get_cluster(i)) for i in range(cluster_count)]

    # compute a histogram using matplotlib
    plt.hist(var)
    plt.show()

    #save plt to file
    plt.savefig('var.png')

