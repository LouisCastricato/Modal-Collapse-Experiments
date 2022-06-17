# this file compares the accuracy per cluster to the AUC

from modalcollapse.utils import *
import matplotlib.pyplot as plt
from functools import partial
from modalcollapse.indexing.faiss_utils import singular_value_plot_faiss, batch
from multiprocess import Pool
from tqdm import tqdm
from multiprocess import Pool
import matplotlib.pyplot as plt

from glob import glob



def compute_scatter(graphs, points, clusters, file_name, title=None):

    accs = []
    aucs = []

    # compute the area under the curve
    for g in graphs:
        aucs.append(np.trapz(g))

    # we'll multithread the accuracy computation
    def map_function(cluster_points):
        cluster, points = cluster_points

        anchors = []
        positives = []
        negatives = []
        for idx, point in enumerate(cluster):
            if type(point) != tuple:
                continue

            # get the anchors and positives
            anchors.append(points[idx])
            positives.append(point[1][0][0])
            negatives.append(point[1][1])

        # the anchors and positives right now are a list of length N of d embeddings. Stack them
        anchors = np.stack(anchors)
        positives = np.stack(positives)
        negatives = np.stack(negatives)

        # concat positives and negatives
        all_points = np.concatenate([positives, negatives], axis=0)

        # compute the accuracy. don't include the hard negative
        return compute_accuracy(np.dot(anchors, all_points.T))

    cluster_points = zip(clusters, points)
    pool = Pool(processes=16)
    accs = pool.map(map_function, cluster_points)
    pool.close()

    # plot a scatter plot of answer acc vs auc and query acc vs auc
    plt.scatter(accs, aucs)

    plt.xlabel("Accuracy")
    plt.ylabel("AUC")
    if title is None:
        plt.title("MS MARCO Accuracy vs AUC")
    else:
        plt.title("MS MARCO Accuracy vs AUC; " + title)

    plt.show()
    plt.savefig(file_name)
    plt.clf()

    return aucs
 

if __name__ == '__main__':

    hardness = "5_0"
    base_dirs = glob("/home/louis_huggingface_co/Varying-Hardness/"+hardness+"_hardness/*")
    print("Loaded " + str(len(base_dirs)) + " base directories")

    aucs = []
    global_aucs = []
    for idx, version in tqdm(enumerate(base_dirs), total=len(base_dirs)):
        # constants
        base_path = version+"/ms_marco_"
        paths = ["anchor_embeddings"]
        variants = [".npy"]

        # use base path, paths, and variants to produce a set of six strings
        total_paths = [base_path + p + v for p in paths for v in variants]

        # generate data
        anchor_dataset = [np.float32(np.load(path)) for path in total_paths]

        
        # load the data that we're gonna be using to compute accuracy 
        paths = ["positive_embeddings"]
        total_paths = [base_path + p + v for p in paths for v in variants]
        positive_dataset = [np.float32(np.load(path)) for path in total_paths]

        paths = ["negative_embeddings"]
        total_paths = [base_path + p + v for p in paths for v in variants]
        negative_dataset = [np.float32(np.load(path)) for path in total_paths]

        # reshape
        d =  anchor_dataset[0].shape[-1]

        anchor_dataset = [np.reshape(a, (-1, d)) for a in anchor_dataset]
        positive_dataset = [np.reshape(p, (-1, 128, d)) for p in positive_dataset]
        negative_dataset = [np.reshape(n, (-1, d)) for n in negative_dataset]

        paired_dataset = [zip(positive_dataset[i], negative_dataset[i]) for i in range(len(anchor_dataset))]
        
        batched = batch(anchor_dataset, data=paired_dataset)

        # make sure that we are returning the clusters
        sv_faiss_clusters = partial(singular_value_plot_faiss, return_clusters=True, points_per_query=64)

        # get singular values
        singular_values = list(map(sv_faiss_clusters, [batched(t) for t in range(len(anchor_dataset))]))

        graphs, points, clusters = zip(*singular_values)
        graphs = np.array(graphs)
        points = np.array(points)

        aucs.append(compute_scatter(graphs[0], points[0], clusters[0],  "DPR_DeCLUTR_"+hardness+"_"+str(idx)+".png", title="DPR Uni-DeCLUTR " + str(idx)))

        # generate the global AUC as well
        singular_values_global = list(map(generate_singular_value_plot, anchor_dataset))

        global_aucs.append(np.mean(np.array([np.trapz(x) for x in singular_values_global])))

    aucs = np.array(aucs)

    # load the accuracy npy
    accuracy = np.load('/home/louis_huggingface_co/Varying-Hardness/accuracies.npy')
    accuracy = np.repeat(np.expand_dims(accuracy, axis=1), aucs.shape[1], axis=1)

    loss = np.load('/home/louis_huggingface_co/Varying-Hardness/losses.npy')
    loss = np.repeat(np.expand_dims(loss, axis=1), aucs.shape[1], axis=1)

    global_aucs = np.repeat(np.expand_dims(global_aucs, axis=1), aucs.shape[1], axis=1) / np.max(global_aucs)

    plot_input = {
        "Intrinsic AUC": aucs,
        "Extrinsic AUC": global_aucs,
        "Accuracy": accuracy,
        "Loss": loss
    }

    plot_confidence_intervals(plot_input, "DPR_DeCLUTR_"+hardness+"_accuracy.png", show=True)


