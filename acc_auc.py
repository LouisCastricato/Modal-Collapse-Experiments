# this file compares the accuracy per cluster to the AUC

from ctypes import alignment
from modalcollapse.utils import *
import matplotlib.pyplot as plt
from functools import partial
from modalcollapse.indexing.faiss_utils import singular_value_plot_faiss, batch
from multiprocess import Pool
from tqdm import tqdm
from multiprocess import Pool
import matplotlib.pyplot as plt

from glob import glob



def compute_alignment(points, clusters):

    alignment = []

    # we'll multithread the alignment computation
    def map_function(cluster_points):
        cluster, pts = cluster_points

        anchors = []
        positives = []
        for idx, point in enumerate(cluster):
            if type(point) != tuple:
                continue

            # get the anchors and positives
            anchors.append(pts[idx])
            positives.append(point[1])

        # the anchors and positives right now are a list of length N of d embeddings. Stack them
        anchors = np.stack(anchors)
        positives = np.stack(positives)

        # compute the alignment. don't include the hard negative
        return generate_alignment_plot(anchors, positives)
        
    cluster_points = zip(clusters, points)
    pool = Pool(processes=16)
    alignment = pool.map(map_function, cluster_points)
    pool.close()

    return alignment
 
def compute_EMD(points, clusters):

    EMDs = []

    # we'll multithread the earth movers distance computation
    def map_function(cluster_points):
        cluster, pts = cluster_points

        anchors = []
        positives = []
        for idx, point in enumerate(cluster):
            if type(point) != tuple:
                continue

            # get the anchors and positives
            anchors.append(pts[idx])
            positives.append(point[1])

        # the anchors and positives right now are a list of length N of d embeddings. Stack them
        anchors = np.stack(anchors)
        positives = np.stack(positives)

        # compute the alignment. don't include the hard negative
        return generate_alignment_plot(anchors, positives)
        
    cluster_points = zip(clusters, points)
    pool = Pool(processes=16)
    EMDs = pool.map(map_function, cluster_points)
    pool.close()

    return EMDs


if __name__ == '__main__':

    base_dirs = "/home/louis_huggingface_co/Varying-Hardness/"
    pos_subdir = "positive_embeddings_3/"
    anchor_subdir = "anchor_embeddings_3/"

    positive_glob = glob(base_dirs + pos_subdir + "*" + ".npy")
    anchor_glob = glob(base_dirs + anchor_subdir + "*" + ".npy")

    print("Loaded " + str(len(positive_glob)) + " base directories")
    print("Loaded " + str(len(anchor_glob)) + " base directories")

    alignment = []
    for idx, pos, anchor in tqdm(zip(range(len(positive_glob)), positive_glob, anchor_glob), total=len(positive_glob)):
        # generate data
        positive_dataset = np.load(pos)
        anchor_dataset = np.load(anchor)
        # reshape to -1, d
        positive_dataset = [positive_dataset.reshape(-1, positive_dataset.shape[-1])]
        anchor_dataset = [anchor_dataset.reshape(-1, anchor_dataset.shape[-1])]

        # use the positives as batched elements        
        batched = batch(anchor_dataset, data=positive_dataset)

        # make sure that we are returning the clusters
        sv_faiss_clusters = partial(singular_value_plot_faiss, return_clusters=True, points_per_query=64)

        # get singular values
        singular_values = list(map(sv_faiss_clusters, [batched(t) for t in range(len(anchor_dataset))]))

        graphs, points, clusters = zip(*singular_values)
        graphs = np.array(graphs)
        points = np.array(points)

        alignment.append(compute_alignment(points[0], clusters[0]))
    # save alignment to a .npy
    np.save("alignment.npy", alignment)


