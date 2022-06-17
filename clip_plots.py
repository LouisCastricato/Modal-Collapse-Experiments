import matplotlib.pyplot as plt
from functools import partial
from modalcollapse.indexing.faiss_utils import distance_to_centroid_faiss, singular_value_plot_faiss, batch
from modalcollapse.indexing.faiss_indexers import DenseFlatIndexer
from modalcollapse.utils import generate_singular_value_plot
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from multiprocess import Pool

from glob import glob
import numpy as np

import pandas as pd 
from scipy.special import softmax


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


if __name__ == '__main__':

    # constants
    clusters_to_keep = 3

    npy_files = glob("/home/louis_huggingface_co/clip-embeddings/**/*.npy", recursive=True)
    npy_files.sort()

    # filter npy file names to those that include either "coco" or "laion2B-en"
    to_save_name = list(map(lambda x: '_'.join(x.split('/')[4:]).split('.')[0], npy_files))

    # generate data
    print("Loading")
    datasets = [np.float32(np.load(path, allow_pickle=True)) for path in npy_files]

    # we're going to do this twice. On the first one, the dataset is composed of all text embeddings (corresponding to even positions) and the data is composed of image embedings
    # firstly take all even elements from dataset
    text_embeddings = datasets[1::2]
    # then take all odd elements from dataset
    image_embeddings = datasets[::2]
    combined = list(map(lambda x: np.concatenate((x[0], x[1]), axis=0), zip(text_embeddings, image_embeddings)))

    print("Done loading")

    text_batched = batch(text_embeddings, data=image_embeddings)
    image_batched = batch(image_embeddings, data=text_embeddings)
    combined_batched = batch(combined)

    sv_plot_faiss_with_return = partial(singular_value_plot_faiss, points_per_query = 768, return_clusters=True)
    sv_plot_faiss = partial(singular_value_plot_faiss, points_per_query = 512)

    # get variance
    #text_singular_values_clusters = list(map(sv_plot_faiss_with_return, [text_batched(t) for t in tqdm(range(len(text_embeddings)))]))
    #image_singular_values_clusters = list(map(sv_plot_faiss_with_return, [image_batched(t) for t in tqdm(range(len(image_embeddings)))]))

    
    singular_values_clusters = list(map(sv_plot_faiss, [combined_batched(t) for t in tqdm(range(len(image_embeddings)))]))
    singular_values_global = list(map(generate_singular_value_plot, tqdm(combined)))


    for idx, (svl, svg) in enumerate(zip(singular_values_clusters, singular_values_global)):
        for s in svl:
            plt.semilogy(s)

        plt.show()
        plt.savefig('graphs_combined/' + to_save_name[::2][idx] + ".png")
        plt.clf()

        plt.semilogy(svg)
        plt.show()
        plt.savefig('graphs_combined/' + to_save_name[::2][idx] + '_global.png')
        plt.clf()
    """
    
    text_r2 = list()
    image_r2 = list()

    # create an array of sv, cluster tuples so that we can sort by auc singular value
    for dataset, name in zip(text_singular_values_clusters, to_save_name[1::2]):
        accuracy = list()
        aucs = list()
        for svs, cluster_points, embeddings in zip(*dataset):

            embedding_matrix = list(map(lambda x: embeddings[x][1], range(len(embeddings))))
            # unsqueeze embedding matrix to make it a 2d array and stack
            embedding_matrix = np.stack(embedding_matrix, axis=0)
            accuracy.append(compute_accuracy((cluster_points @ np.transpose(embedding_matrix))))
            aucs.append(np.trapz(svs))

        # compute r2 for aucs vs auccracy
        r2 = np.corrcoef(aucs, accuracy)[0, 1] ** 2
        text_r2.append(r2)

        plt.scatter(x = aucs, y = accuracy)
        plt.savefig('accuracy_graphs/'+ name + '_accuracy.png')
        plt.clf()

    # create an array of sv, cluster tuples so that we can sort by auc singular value
    for dataset, name in zip(image_singular_values_clusters, to_save_name[::2]):
        accuracy = list()
        aucs = list()
        for svs, cluster_points, embeddings in zip(*dataset):

            embedding_matrix = list(map(lambda x: embeddings[x][1], range(len(embeddings))))
            # unsqueeze embedding matrix to make it a 2d array and stack
            embedding_matrix = np.stack(embedding_matrix, axis=0)
            accuracy.append(compute_accuracy((cluster_points @ np.transpose(embedding_matrix))))
            aucs.append(np.trapz(svs))

        # compute r2 for aucs vs auccracy
        r2 = np.corrcoef(aucs, accuracy)[0, 1] ** 2
        image_r2.append(r2)

        plt.scatter(x = aucs, y = accuracy)
        plt.savefig('accuracy_graphs/'+ name + '_accuracy.png')
        plt.clf()
    
    # box and whisker plot for text r2 and image r2
    plt.boxplot([text_r2, image_r2])
    plt.xticks([1, 2], ['Text', 'Image'])
    plt.title('R2 AUC vs Top-1 Retrieval Accuracy for Text and Image Embeddings')
    plt.savefig('accuracy_graphs/r2_boxplot.png')
    """