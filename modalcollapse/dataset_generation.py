import numpy as np
from power_spherical import HypersphericalUniform, MarginalTDistribution, PowerSpherical
from scipy.stats import ortho_group
import skdim
import torch


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

def generate_power_spherical(points_per_cluster=100, dim=512, clusters=200, scale_upper_bound=1.0):
    """
    A more advanced version of get_splooch_points.
    :param points_per_cluster: number of points to generate per cluster
    :param dim: dimension of the hypersphere
    :param clusters: number of clusters
    :param scale_upper_bound: upper bound on the scale of the hypersphere
    :return: numpy array of shape (points_per_cluster * clusters, dim)
    """

    # first, for every cluster, generate a normalized vector of dimension dim
    # we can do this by gneerating using randn and normalizing
    cluster_vectors = np.array([np.random.randn(dim) for _ in range(clusters)])
    cluster_vectors = torch.tensor(cluster_vectors / np.linalg.norm(cluster_vectors, axis=1).reshape(-1, 1))

    # we need to generate scale scalars that max out at scale_upper_bound
    # firstly generate an arbitrarily small episilon
    epsilon = np.random.uniform(min(scale_upper_bound/100, 1e-6), max(scale_upper_bound/100, 1e-6))
    scale = torch.tensor(np.random.uniform(epsilon, scale_upper_bound, clusters))

    points = list()
    for vec, s in zip(cluster_vectors, scale):
        dist = PowerSpherical(vec, s)
        points.append(dist.sample((points_per_cluster,)))

    return np.concatenate(points)

# testing
if __name__ == "__main__":

    dataset_A = generate_power_spherical()
    dataset_B = generate_power_spherical()

    # save dataset A and dataset B
    np.save("dataset_A.npy", dataset_A)
    np.save("dataset_B.npy", dataset_B)


