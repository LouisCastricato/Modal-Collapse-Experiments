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
from glob import glob 

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


def compute_accuracy_auc(graphs, clusters, dataset, file_name, title=None):

    acc_answers = []
    acc_queries = []
    acc_answers_queries = []
    aucs = []

    # used for extrinsic acc
    all_queries = []
    all_answers = []

    for i, cluster in enumerate(clusters):
        # used for intrinsic acc
        all_pts_cluster = []
        all_answers_cluster = []
        all_queries_cluster = []
        for point in cluster:
            if type(point) != tuple:
                continue

            # point[0] refers to an index in the original dataset
            all_pts_cluster.append(dataset[point[0]])

            # the other two are the answer and query embeddings
            all_answers_cluster.append(point[1][0])
            all_queries_cluster.append(point[1][1])

        all_pts_cluster = np.stack(all_pts_cluster, axis=0)
        all_answers_cluster = np.stack(all_answers_cluster, axis=0)
        all_queries_cluster = np.stack(all_queries_cluster, axis=0)


        acc_answers.append(compute_accuracy((all_pts_cluster @ np.transpose(all_answers_cluster))))
        acc_queries.append(compute_accuracy((all_pts_cluster @ np.transpose(all_queries_cluster)))) 
        acc_answers_queries.append(compute_accuracy((all_answers_cluster @ np.transpose(all_queries_cluster))))
        aucs.append(np.trapz(graphs[i]))

        # recall that answers cluster and query cluster are duplicated in the same fashion
        all_answers_cluster, index = np.unique(all_answers_cluster, axis=0, return_index=True)
        all_queries_cluster = all_queries_cluster[index]

        all_answers.append(all_answers_cluster)
        all_queries.append(all_queries_cluster)

    # stack all the points together
    all_answers = np.concatenate(all_answers, axis=0)
    all_queries = np.concatenate(all_queries, axis=0)

    # compute accs
    # convert everything to fp16
    length = all_answers.shape[0]
    # generate 1k samples from range(length)
    indices = np.random.choice(length, size=1000, replace=False)

    all_answers = all_answers.astype(np.float16)[indices]
    all_queries = all_queries.astype(np.float16)[indices]
    acc_answers_queries_global = compute_accuracy((all_answers @ np.transpose(all_queries)))

    # return accs and aucs
    return acc_answers, acc_queries, acc_answers_queries, aucs, acc_answers_queries_global

 
# plots a box and whisker plot
def plot_box_and_whisker(aucs, title=None):
    plt.figure(figsize=(10, 10))
    plt.boxplot([acc_answers, acc_queries, acc_answers_queries, aucs], labels=['acc_answers', 'acc_queries', 'acc_answers_queries', 'AUC'])
    plt.title(title)
    plt.show()
if __name__ == '__main__':

    hardness = "5_0"
    acc_answers_total = []
    acc_queries_total = []
    acc_answers_queries_total = []
    aucs_total = []

    acc_answers_queries_global_total = []

    base_dirs = glob("/home/louis_huggingface_co/Varying-Hardness/"+hardness+"_hardness/*")
    for version in base_dirs:
        # constants
        base_path = version+"/ms_marco_"
        paths = ["passage_embeddings"]
        variants = ["_v1.npy"]

        # use base path, paths, and variants to produce a set of six strings
        total_paths = [base_path + p + v for p in paths for v in variants]

        # generate data
        print("Loading")
        try:
            base_datasets = [np.float32(np.load(path)) for path in total_paths]
        except:
            print("Failed to load")
            continue
        
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

        graphs, _, clusters = zip(*singular_values)
        graphs = np.array(graphs)


        print(graphs.shape)
        import sys
        sys.exit()
        
        acc_answers, acc_queries, acc_answers_queries, aucs, acc_answers_queries_global\
        = compute_accuracy_auc(graphs[0], clusters[0], base_datasets[0],  "DPR_DeCLUTR_"+hardness+"_"+version+".png", title="DPR Uni-DeCLUTR " + version)

        # append to total
        acc_answers_total.append(acc_answers)
        acc_queries_total.append(acc_queries)
        acc_answers_queries_total.append(acc_answers_queries)
        aucs_total.append(aucs)

        acc_answers_queries_global_total.append(acc_answers_queries_global)


    # convert to npy
    acc_answers_total = np.array(acc_answers_total)
    acc_queries_total = np.array(acc_queries_total)
    acc_answers_queries_total = np.array(acc_answers_queries_total)
    aucs_total = np.array(aucs_total)

    acc_answers_queries_global_total = np.array(acc_answers_queries_global_total)


    # save
    np.save("acc_answers_total_"+hardness+".npy", acc_answers_total)
    np.save("acc_queries_total_"+hardness+".npy", acc_queries_total)
    np.save("acc_answers_queries_total_"+hardness+".npy", acc_answers_queries_total)
    np.save("aucs_total_"+hardness+".npy", aucs_total)
    np.save("acc_answers_queries_global_total_"+hardness+".npy", acc_answers_queries_global_total)

