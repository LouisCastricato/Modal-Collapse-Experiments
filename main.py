from modalcollapse.utils import *
import matplotlib.pyplot as plt
from functools import partial
from modalcollapse.indexing.faiss_utils import distance_to_centroid_faiss, singular_value_plot_faiss, batch
from modalcollapse.indexing.faiss_indexers import DenseFlatIndexer
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from multiprocess import Pool


if __name__ == '__main__':

    # use base path, paths, and variants to produce a set of six strings
    total_paths = ["dataset_A.npy", "dataset_B.npy"]

    # generate data
    print("Loading")
    datasets = [np.float32(np.load(path)) for path in tqdm(total_paths)]
    # for every item in datasets, remove duplicates
    print("Deduplicating")
    datasets = [np.unique(d, axis=0) for d in tqdm(datasets)]

    batched = batch(datasets)

    def cosine_filter_condition(pt1, pt2):
        return (np.dot(pt1, pt2) / (np.linalg.norm(pt1) * np.linalg.norm(pt2)) > 0)

    sv_plot_faiss_filtered = partial(singular_value_plot_faiss)
    # get variance
    singular_values = list(map(sv_plot_faiss_filtered, [batched(t) for t in tqdm(range(len(datasets)))]))
    singular_values_global = list(map(generate_singular_value_plot, tqdm(datasets)))

    for idx, (svl, svg) in enumerate(zip(singular_values, singular_values_global)):
        for s in svl:
            plt.semilogy(s)

        plt.show()
        plt.savefig('graphs/singular_values_batch_' + str(idx) + '.png')
        plt.clf()

        plt.semilogy(svg)
        plt.show()
        plt.savefig('graphs/singular_values_global_' + str(idx) + '.png')
        plt.clf()


