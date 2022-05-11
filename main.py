from modalcollapse.utils import *
import matplotlib.pyplot as plt
from functools import partial
from modalcollapse.indexing.faiss_utils import distance_to_centroid_faiss, singular_value_plot_faiss, batch
from modalcollapse.indexing.faiss_indexers import DenseFlatIndexer
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from multiprocess import Pool


if __name__ == '__main__':

    # constants
    data_points = 1000000
    dim = 128
    cluster_count = 100
    rotation_count = 1
    base_path = "/mnt/raid/gits/tda-CLIP/cloob_laion_400m_vit_b_16_32_epochs_coco_train2017_embeds/cloob_laion_400m_vit_b_16_32_epochs_coco_train2017_" 
    path1 = base_path + "image_embeds.npy"
    path2 = base_path + "text_embeds.npy"
    # generate data
    image_embeds = np.load(path1)
    text_embeds = np.load(path2)
    both_embeds = np.concatenate((image_embeds, text_embeds), axis=0)
    datasets = [image_embeds, text_embeds, both_embeds]

    batched = batch(datasets)

    def cosine_filter_condition(pt1, pt2):
        return (np.dot(pt1, pt2) / (np.linalg.norm(pt1) * np.linalg.norm(pt2)) > 0.2)

    # get variance
    variances = list(map(partial(distance_to_centroid_faiss, filter_condition=cosine_filter_condition), [batched(t) for t in tqdm(range(len(datasets)))]))
    singular_values = list(map(singular_value_plot_faiss, [batched(t) for t in tqdm(range(len(datasets)))]))
    singular_values_global = list(map(generate_singular_value_plot, tqdm(datasets)))

    for idx, (var, svl, svg) in enumerate(zip(variances, singular_values, singular_values_global)):
        # compute a histogram using matplotlib
        print("Skew, kurtosis:", skew(var), kurtosis(var))
        plt.hist(var, bins=64)
        plt.show()
        plt.savefig('graphs/variance_batch_' + str(idx) + '.png')
        plt.clf()

        for s in svl:
            plt.semilogy(s)

        plt.show()
        plt.savefig('graphs/singular_values_batch_' + str(idx) + '.png')
        plt.clf()

        plt.semilogy(svg)
        plt.show()
        plt.savefig('graphs/singular_values_global_' + str(idx) + '.png')
        plt.clf()


