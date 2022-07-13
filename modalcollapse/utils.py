# implements a number of helper functions to compute k-means clusters and related
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from sklearn.cluster import KMeans
from scipy.special import softmax
from scipy.signal import savgol_filter
from scipy.spatial.distance import pdist

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


def generate_singular_value_plot(data, size=10000):
    """
    Generates a singular value plot for the given data.
    :param data: numpy array of shape (n, d)
    :param size: number of points to use for the plot
    """
    # if len(data) > size, sample data
    if len(data) > size:
        idxs = np.random.choice(range(len(data)), size, replace=False)
        data = data[idxs]
    data = np.float64(data)
    # compute SVD
    try:
        return np.flip(np.linalg.eigvalsh(np.cov(data.T)))
    except:
        return np.ones(data.shape[1])

def generate_uniformity_plot(data):
    """
    Generates a uniformity plot for the given data.
    :param data: numpy array of shape (n, d)
    """
    # compute SVD
    try:
        sq_pdist = pdist(data.T, 'sqeuclidean')
        return np.log(np.mean(np.exp(-2 * sq_pdist)))
    except:
        return np.ones(data.shape[1])


def compute_accuracy(contrastive_matrix):
    """
    :param contrastive_matrix: an nxn matrix
    :return: accuracy (scalar from 0 to 1)
    """
    contrastive_matrix_i = np.argmax(softmax(contrastive_matrix, axis=0), axis=0).tolist()
    contrastive_matrix_j = np.argmax(softmax(contrastive_matrix, axis=1), axis=1).tolist()

    labels = list(range(contrastive_matrix.shape[1]))
    acc_i = np.mean(np.array([contrastive_matrix_i[i] == labels[i] for i in range(len(labels))]))
    acc_j = np.mean(np.array([contrastive_matrix_j[i] == labels[i] for i in range(len(labels))]))
    
    return (acc_i + acc_j) / 2.


# returns the AUC of min(get_intrinsic_dimension_plot(datasets))
def plot_scatter(x,y):
    plt.scatter(x,y)
    plt.show()
    plt.clf()

def plot_confidence_intervals(plts, save_to_dir = None, title = None, show=True, x=None, xlabel=None, ylabel=None, smooth=False, subplot=False):
    """
    Takes a dict of 2d numpy array and creates a confidence intv plot for every column
    :param x: a dict of 2d numpy array
    :param save_to_dir: directory to save the plot to
    :param title: title of the plot
    :param show: whether to show the plot
    :param x: optional x values
    :param xlabel: optional x axis label
    :param ylabel: optional y axis label
    :param smooth: whether to smooth the plot
    :param subplot: whether to plot in a subplot
    :return: None
    """
    plt.figure(figsize=(20, 10))
    colors = []
    if subplot:
        subplot_len = len(list(plts.values()))

    for idx, y in enumerate(plts.values()):

        if subplot:
            plt.subplot(1, subplot_len, idx+1)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(list(plts.keys())[idx])

        # convert x to a list of numpy arrays
        means = np.min(y, axis=1)

        if smooth:
            means = savgol_filter(means, 13, 5)

        if x is None:
            colors.append(plt.plot(means).pop(0).get_color())
        else:
            colors.append(plt.plot(x, means).pop(0).get_color())

        if min(y[0]) == max(y[0]):
            continue
        ci = 1.96 * np.std(y)/np.sqrt(len(y))
        if x is None:
            plt.fill_between(range(len(y)), means - ci, means + ci, alpha=0.5)
        else:
            plt.fill_between(x, means - ci, means + ci, alpha=0.5)

    if not subplot:
        # set axis labels
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)

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