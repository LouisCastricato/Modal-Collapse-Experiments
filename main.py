from utils import *
import matplotlib.pyplot as plt
from functools import partial
from indexing.faiss_utils import distance_to_centroid_faiss, singular_value_plot_faiss, batch
from indexing.faiss_indexers import DenseFlatIndexer
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from multiprocess import Pool

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
    base_path = "/mnt/raid/gits/tda-CLIP/cloob_laion_400m_vit_b_16_32_epochs_coco_train2017_embeds/cloob_laion_400m_vit_b_16_32_epochs_coco_train2017_" 
    path1 = base_path + "image_embeds.npy"
    path2 = base_path + "text_embeds.npy"
    # generate data
    image_embeds = np.load(path1)
    text_embeds = np.load(path2)
    both_embeds = np.concatenate((image_embeds, text_embeds), axis=0)
    datasets = [image_embeds, text_embeds, both_embeds]

    batched = batch(datasets)

    def cosine_filter_condition(pt1, pt2):
        return (np.dot(pt1, pt2) / (np.linalg.norm(pt1) * np.linalg.norm(pt2)) > 0.2)

    # get variance
    variances = list(map(partial(distance_to_centroid_faiss, filter_condition=cosine_filter_condition), [batched(t) for t in tqdm(range(len(datasets)))]))
    singular_values = list(map(singular_value_plot_faiss, [batched(t) for t in tqdm(range(len(datasets)))]))
    singular_values_global = list(map(generate_singular_value_plot, tqdm(datasets)))

    for idx, (var, svl, svg) in enumerate(zip(variances, singular_values, singular_values_global)):
        # compute a histogram using matplotlib
        print("Skew, kurtosis:", skew(var), kurtosis(var))
        plt.hist(var, bins=64)
        plt.show()
        plt.savefig('graphs/variance_batch_' + str(idx) + '.png')
        plt.clf()

        for s in svl:
            plt.semilogy(s)

        plt.show()
        plt.savefig('graphs/singular_values_batch_' + str(idx) + '.png')
        plt.clf()

        plt.semilogy(svg)
        plt.show()
        plt.savefig('graphs/singular_values_global_' + str(idx) + '.png')
        plt.clf()


