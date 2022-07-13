from pickle import NONE
import numpy as np
from power_spherical import HypersphericalUniform, MarginalTDistribution, PowerSpherical
from scipy.stats import ortho_group
import skdim
import torch
from tqdm import tqdm

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

def generate_power_spherical(points_per_cluster=100, dim=512, clusters=100, scale_upper_bound=100.0, scale_lower_bound=None, cluster_vectors=None):
    """
    A more advanced version of get_splooch_points.
    :param points_per_cluster: number of points to generate per cluster
    :param dim: dimension of the hypersphere
    :param clusters: number of clusters
    :param scale_upper_bound: upper bound on the scale of the hypersphere
    :param scale_lower_bound: lower bound on the scale of the hypersphere
    :param cluster_vectors: cluster vectors
    :return: numpy array of shape (points_per_cluster * clusters, dim)
    """

    # first, for every cluster, generate a normalized vector of dimension dim
    # we can do this by gneerating using randn and normalizing
    if cluster_vectors is None:
        cluster_vectors = np.array([np.random.randn(dim) for _ in range(clusters)])
        cluster_vectors = torch.tensor(cluster_vectors / np.linalg.norm(cluster_vectors, axis=1).reshape(-1, 1))

    # we need to generate scale scalars that max out at scale_upper_bound
    # firstly generate an arbitrarily small episilon
    if scale_lower_bound is None:
        epsilon = np.random.uniform(min(scale_upper_bound/100, 1e-6), max(scale_upper_bound/100, 1e-6))
    else:
        epsilon = scale_lower_bound
    scale = torch.tensor(np.random.uniform(epsilon, scale_upper_bound, clusters))

    points = list()
    for vec, s in zip(cluster_vectors, scale):
        dist = PowerSpherical(vec, s)
        points.append(dist.sample((points_per_cluster,)))

    return np.concatenate(points)

# generate synthetic dataset
if __name__ == "__main__":

    # iterate over the number of clusters and use that to create a plot
    amts = list(range(100, 500, 25))
    datasets_A = []
    datasets_B = []
    scales = np.linspace(100., 1000., 20)

    # initialize cluster vectors
    cluster_vectors = np.array([np.random.randn(75) for _ in range(100)])
    cluster_vectors = torch.tensor(cluster_vectors / np.linalg.norm(cluster_vectors, axis=1).reshape(-1, 1))

    for scale in tqdm(scales):
        datasets_A.append(np.float32(generate_power_spherical(dim=75, points_per_cluster=250,
        scale_upper_bound=scale, scale_lower_bound=scale, cluster_vectors=cluster_vectors)))

    # save the datasets
    # go through zip(dataset_A, dataset_B) and save it to a file where the clusters number is the filename
    for i, dataset in enumerate(datasets_A):
        np.save(f"../known-intrinsic-dim/datasets_A/A_{int(scales[i])}.npy", dataset)
    
    
    from modalcollapse.indexing.faiss_utils import singular_value_plot_faiss, batch

    batched = batch(datasets_A)
    singular_values = list(map(singular_value_plot_faiss, tqdm([batched(t) for t in range(len(datasets_A))])))
    
    # for every singular value graph, compute the AUC
    all_auc = list()
    for dataset in singular_values:
        dataset_auc = list()
        for graph in dataset:
            dataset_auc.append(np.trapz(graph))
        all_auc.append(dataset_auc)


    all_auc = np.array(all_auc)
    from utils import plot_confidence_intervals

    plot_dict = {
        "Intrinsic AUC (Mean)": all_auc,
    }
    plot_confidence_intervals(plot_dict, x=np.array(scales),
        save_to_dir="scale_vs_auc.png", title="Scale vs AUC",
        xlabel="Scale", ylabel="AUC")


