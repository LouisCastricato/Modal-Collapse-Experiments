# implements a number of helper functions to compute k-means clusters and related
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import ortho_group
from scipy.special import softmax
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
    return np.linalg.norm(data - centroid, axis=1)

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


def generate_singular_value_plot(data, k=None, size=10000):
    """
    Generates a singular value plot for the given data.
    :param data: numpy array of shape (n, d)
    :param k: number of singular values to plot
    :param size: number of points to udse for the plot
    """
    # if len(data) > size, sample data
    if len(data) > size:
        idxs = np.random.choice(range(len(data)), size, replace=False)
        data = data[idxs]
    data = np.float64(data)
    # compute SVD
    return np.flip(np.linalg.eigvalsh(np.cov(data.T)))

def compute_accuracy(contrastive_matrix):
    """
    :param contrastive_matrix: an nxn matrix
    :return: accuracy (scalar from 0 to 1)
    """
    contrastive_matrix_i = np.argmax(softmax(contrastive_matrix, axis=0), axis=0).tolist()
    contrastive_matrix_j = np.argmax(softmax(contrastive_matrix, axis=1), axis=1).tolist()

    labels = list(range(contrastive_matrix.shape[0]))
    acc_i = np.mean(np.array([contrastive_matrix_i[i] == labels[i] for i in range(len(labels))]))
    acc_j = np.mean(np.array([contrastive_matrix_j[i] == labels[i] for i in range(len(labels))]))

    return (acc_i + acc_j) / 2.


# returns the AUC of min(get_intrinsic_dimension_plot(datasets))
def plot_scatter(x,y):
    plt.scatter(x,y)
    plt.show()
    plt.clf()

def plot_confidence_intervals(plts, save_to_dir = None, title = None, show=True):
    """
    Takes a dict of 2d numpy array and creates a confidence intv plot for every column
    :param x: a dict of 2d numpy array
    :param save_to_dir: directory to save the plot to
    :param title: title of the plot
    :param show: whether to show the plot
    :return: None
    """
    plt.figure(figsize=(10, 10))
    colors = []
    for x in plts.values():
        # convert x to a list of numpy arrays
        means = x[:,0]
        colors.append(plt.plot(means).pop(0).get_color())

        if min(x[0]) == max(x[0]):
            continue
        ci = 1.96 * np.std(x)/np.sqrt(len(x))
        plt.fill_between(range(len(x)), means - ci, means + ci, alpha=0.5)

    # set title
    if title is not None:
        plt.title(title)

    # set key
    legend_plots = []
    for c in colors:
        legend_plots.append(mlines.Line2D([], [], color=c, linestyle='-', linewidth=2))
    plt.legend(legend_plots, plts.keys())

    if show:
        plt.show()

    # save
    if save_to_dir is not None:
        plt.savefig(save_to_dir)