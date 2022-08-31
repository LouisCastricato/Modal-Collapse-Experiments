from modalcollapse.utils import *
import matplotlib.pyplot as plt
from functools import partial
from modalcollapse.indexing.faiss_utils import distance_to_centroid_faiss, singular_value_plot_faiss, batch, uniformity_plot_faiss
from modalcollapse.indexing.faiss_indexers import DenseFlatIndexer
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from multiprocess import Pool

from glob import glob

epsilon = 1e-6

if __name__ == '__main__':


    base_dirs = "/home/louis_huggingface_co/Varying-Hardness/"
    pos_subdir = "positive_embeddings_3/"
    anchor_subdir = "anchor_embeddings_3/"

    # use base path, paths, and variants to produce a set of six strings
    total_paths = glob(base_dirs + pos_subdir + "*.npy")[:100]
    anchors_paths = glob(base_dirs + anchor_subdir + "*.npy")[:100]

    def get_number(str):
        return float('.'.join(str.split("/")[-1].split(".")[:-1]))
    scores= list(map(get_number, total_paths))

    # sort total_paths and anchors_paths by score
    total_paths = [total_paths[i] for i in np.argsort(scores)]
    anchors_paths = [anchors_paths[i] for i in np.argsort(scores)]

    # generate data
    print("Loading")
    datasets = [np.float32(np.load(path)) for path in tqdm(total_paths)]
    anchor_datasets = [np.float32(np.load(path)) for path in anchors_paths]

    # for every element of datasets and anchor datasets, reshape to -1,d
    datasets = [datasets[i].reshape(-1, datasets[i].shape[-1]) for i in range(len(datasets))]
    anchor_datasets = [anchor_datasets[i].reshape(-1, anchor_datasets[i].shape[-1]) for i in range(len(anchor_datasets))]

    print("Constructing FAISS index.")
    batched = batch(datasets)
    faiss_batched = [batched(t) for t in range(len(datasets))]

    def cosine_filter_condition(pt1, pt2):
        return (np.dot(pt1, pt2) / (np.linalg.norm(pt1) * np.linalg.norm(pt2)) > 0)

    uniformity_plot_faiss_filtered = partial(uniformity_plot_faiss, points_per_query=300)
    sv_plot_faiss_filtered = partial(singular_value_plot_faiss, points_per_query=300)


    # get variance
    print("Intrinsic Dimension...")
    singular_values = list(map(sv_plot_faiss_filtered, tqdm(faiss_batched)))
    print("Uniformity...")
    uniformity_values = list(map(uniformity_plot_faiss_filtered, tqdm(faiss_batched)))
    print("Global Uniformity...")
    uniformity_values_global = list(map(generate_uniformity_plot, tqdm(datasets)))

    aucs = list()
    uniformity = list()
    for idx, (svl, uni) in enumerate(zip(singular_values, uniformity_values)):
        intrinsic_auc = list()
        intrinsic_uni = list()
        for s, u in zip(svl, uni):
            intrinsic_auc.append(np.trapz(s))
            intrinsic_uni.append(u)
        aucs.append(intrinsic_auc)
        uniformity.append(intrinsic_uni)

    loss = np.load("/home/louis_huggingface_co/translation_rag/translation-rag/RAG_losses.npy")
    validation = np.load("/home/louis_huggingface_co/Varying-Hardness/losses_3.npy")
    validation_acc = np.load("/home/louis_huggingface_co/Varying-Hardness/accuracies_3.npy")

    # comptue extrinsic alignment
    alignments=[]
    for pos, anchor in zip(datasets, anchor_datasets):
        alignments.append(generate_alignment_plot(pos, anchor))
    alignments = np.array(alignments)
    intrinsic_alignment = np.load("/home/louis_huggingface_co/Modal-Collapse-Experiments/alignment.npy")
    intrinsic_alignment = [intrinsic_alignment[i] for i in np.argsort(scores)]


    # aucs is time x N, validation is time x 1. Duplicate the validation vector to match the time dimension of aucs
    loss = np.repeat(np.expand_dims(loss, axis=1), len(aucs[0]), axis=1)
    validation = np.repeat(np.expand_dims(validation, axis=1), len(aucs[0]), axis=1)
    validation_acc = np.repeat(np.expand_dims(validation_acc, axis=1), len(aucs[0]), axis=1)
    uniformity_values_global = np.repeat(np.expand_dims(np.array(uniformity_values_global), axis=1), len(aucs[0]), axis=1)
    alignments = np.repeat(np.expand_dims(alignments, axis=1), len(aucs[0]), axis=1)

    # subtract the last value from all other values
    #for idx, auc in enumerate(aucs):
    #    aucs[idx] = auc - auc[-1]
    #for idx, auc in enumerate(uniformity):
    #    uniformity[idx] = auc - auc[-1]
    #for idx, auc in enumerate(uniformity_values_global):
    #    uniformity_values_global[idx] = auc - uniformity_values_global[-1]
    #for idx, auc in enumerate(alignments):
    #    alignments[idx] = auc - alignments[-1]
    #for idx, auc in enumerate(intrinsic_alignment):
    #    intrinsic_alignment[idx] = auc - auc[-1]
    

    ci_dict = {
        "Intrinsic AUC" :  np.array(aucs)[:100],
        "Intrinsic Uniformity" : np.array(uniformity)[:100],
        "Extrinsic Uniformity" :  np.array(uniformity_values_global)[:100],
        "Intrinsic Alignment" :  np.array(intrinsic_alignment)[:100],
        "Extrinsic Alignment" :  np.array(alignments)[:100],
        #"Loss": np.array(loss),
        "Validation Loss" : validation[:100],
        "Validation Accuracy" : validation_acc[:100],
    }
    x= list(range(0, 100))
    # plot confidence interval
    plot_confidence_intervals(ci_dict, 
    "aucs_vs_time.png", 
    "Collape vs Time",
    xlabel="Time", ylabel="Collapse Metric",
    x=x, subplot=True, smooth=False)

