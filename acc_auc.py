# this file compares the accuracy per cluster to the AUC

from modalcollapse.utils import *
import matplotlib.pyplot as plt
from functools import partial
from modalcollapse.indexing.faiss_utils import distance_to_centroid_faiss, singular_value_plot_faiss, batch
from modalcollapse.indexing.faiss_indexers import DenseFlatIndexer
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from multiprocess import Pool
from scipy.special import softmax
import matplotlib.pyplot as plt

def compute_accuracy(contrastive_matrix):
    """
    :param contrastive_matrix: an nxn matrix
    :return: accuracy (scalar from 0 to 1)
    """
    contrastive_matrix_i = np.argmax(softmax(contrastive_matrix, axis=0), axis=0).tolist()
    contrastive_matrix_j = np.argmax(softmax(contrastive_matrix, axis=1), axis=1).tolist()

    labels = list(range(contrastive_matrix.shape[1]))
    acc_i = np.mean([contrastive_matrix_i[i] == labels[i] for i in range(len(labels))])
    acc_j = np.mean([contrastive_matrix_j[i] == labels[i] for i in range(len(labels))])

    return (acc_i + acc_j) / 2.


# returns the AUC of min(get_intrinsic_dimension_plot(datasets))
def plot_scatter(x,y):
    plt.scatter(x,y)
    plt.show()
    plt.clf()

def compute_scatter(graphs, clusters, dataset, file_name, title=None):

    acc_answers = []
    acc_queries = []
    acc_answers_queries = []
    aucs = []
    for i, cluster in enumerate(clusters):
        all_pts = []
        all_answers = []
        all_queries = []
        for point in cluster:
            if type(point) != tuple:
                continue

            # point[0] refers to an index in the original dataset
            all_pts.append(dataset[point[0]])

            # the other two are the answer and query embeddings
            all_answers.append(point[1][0])
            all_queries.append(point[1][1])

        all_pts = np.stack(all_pts, axis=0)
        all_answers = np.stack(all_answers, axis=0)
        all_queries = np.stack(all_queries, axis=0)


        acc_answers.append(compute_accuracy((all_pts @ np.transpose(all_answers))))
        acc_queries.append(compute_accuracy((all_pts @ np.transpose(all_queries)))) 
        acc_answers_queries.append(compute_accuracy((all_answers @ np.transpose(all_queries))))
        aucs.append(np.trapz(graphs[i]))

    # plot a scatter plot of answer acc vs auc and query acc vs auc
    plt.scatter(acc_answers, aucs)
    plt.scatter(acc_queries, aucs)
    plt.scatter(acc_answers_queries, aucs)

    plt.xlabel("Accuracy")
    plt.ylabel("AUC")
    if title is None:
        plt.title("MS MARCO Accuracy vs AUC")
    else:
        plt.title("MS MARCO Accuracy vs AUC; " + title)

    plt.legend(["Passage <-> Answer", "Passage <-> Query", "Answer <-> Query"])

    plt.show()
    plt.savefig(file_name)
    plt.clf()

 

if __name__ == '__main__':

    # constants
    base_path = "/home/louis_huggingface_co/Modal-Collapse-Experiments/ms_marco_"
    paths = ["passage_embeddings"]
    variants = [".npy", "_v1.npy"]

    # use base path, paths, and variants to produce a set of six strings
    total_paths = [base_path + p + v for p in paths for v in variants]

    # generate data
    print("Loading")
    base_datasets = [np.float32(np.load(path)) for path in total_paths]
    
    # load the data that we're gonna be using to compute accuracy 
    paths = ["answer_embeddings"]
    total_paths = [base_path + p + v for p in paths for v in variants]
    answer_dataset = [np.float32(np.load(path)) for path in total_paths]

    paths = ["query_embeddings"]
    total_paths = [base_path + p + v for p in paths for v in variants]
    query_dataset = [np.float32(np.load(path)) for path in total_paths]

    paired_dataset = [zip(answer_dataset[i], query_dataset[i]) for i in range(len(answer_dataset))]
    
    batched = batch(base_datasets, data=paired_dataset)

    # make sure that we are returning the clusters
    sv_faiss_clusters = partial(singular_value_plot_faiss, return_clusters=True)

    # get singular values
    singular_values = list(map(sv_faiss_clusters, [batched(t) for t in tqdm(range(len(base_datasets)))]))

    graphs, clusters = zip(*singular_values)
    graphs = np.array(graphs)

    compute_scatter(graphs[0], clusters[0], base_datasets[0], "v2_scatter.png", title="mpnet v2")
    compute_scatter(graphs[1], clusters[1], base_datasets[1], "v1_scatter.png", title="mpnet v1")
