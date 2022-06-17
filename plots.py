import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
# loads aucs_total_hardness.npy and plots box plots

def plot_box_plot(plts, save_to_dir = None, title = None):
    """
    Takes a dict of 2d numpy array and creates a boxplot for every column
    :param x: a dict of 2d numpy array
    :param save_to_dir: directory to save the plot to
    :return: None
    """
    plt.figure(figsize=(10, 10))
    for x in plts.values():
        # convert x to a list of numpy arrays
        x = x.tolist()
        # set the color to something random
        plt.boxplot(x)

    # set title
    if title is not None:
        plt.title(title)

    # set key
    plt.legend(plts.keys())

    plt.show()

    # save
    if save_to_dir is not None:
        plt.savefig(save_to_dir)

def plot_confidence_intervals(plts, save_to_dir = None, title = None, show=True):
    """
    Takes a dict of 2d numpy array and creates a confidence intv plot for every column
    :param x: a dict of 2d numpy array
    :param save_to_dir: directory to save the plot to
    :param title: title of the plot
    :param show: whether to show the plot
    :return: None
    """
    plt.figure(figsize=(10, 10))
    colors = []
    for x in plts.values():
        # convert x to a list of numpy arrays
        means = np.mean(x, axis=0)
        colors.append(plt.plot(means).pop(0).get_color())

        ci = 1.96 * np.std(x)/np.sqrt(len(x))
        plt.fill_between(range(len(x)), means - ci, means + ci, alpha=0.5)

    # set title
    if title is not None:
        plt.title(title)

    # set key
    legend_plots = []
    for c in colors:
        legend_plots.append(mlines.Line2D([], [], color=c, linestyle='-', linewidth=2))
    plt.legend(legend_plots, plts.keys())

    if show:
        plt.show()

    # save
    if save_to_dir is not None:
        plt.savefig(save_to_dir)
# rather than do a box plot, for each column average the values
def plot_averages(plts, save_to_dir = None, title = None):
    """
    Takes a dict of 2d numpy array and plots the average of each column
    :param x: a dict of 2d numpy array
    :param save_to_dir: directory to save the plot to
    :return: None
    """
    plt.figure(figsize=(10, 10))
    for x in plts.values():
        # convert x to a list of numpy arrays
        x = x.tolist()

        plt.plot(list(map(np.mean, x)))

    # set title
    if title is not None:
        plt.title(title)

    # set key
    plt.legend(plts.keys())

    plt.show()

    # save
    if save_to_dir is not None:
        plt.savefig(save_to_dir)


if __name__ == '__main__':
    version = "5_0"
    aucs_file_name = "aucs_total_" + version + ".npy"
    acc_queries_file_name = "acc_queries_total_" + version + ".npy"
    acc_answers_file_name = "acc_answers_total_" + version + ".npy"
    acc_answers_queries_file_name = "acc_answers_queries_total_" + version + ".npy"
    acc_answers_queries_global_total = "acc_answers_queries_global_total_"+version+".npy"

    # load 
    aucs = np.load(aucs_file_name)
    acc_queries = np.load(acc_queries_file_name)
    acc_answers = np.load(acc_answers_file_name)
    acc_answers_queries = np.load(acc_answers_queries_file_name)
    acc_answers_queries_global_total = np.load(acc_answers_queries_global_total)

    # duplicate acc_answers_queries_global_total so its the same size as acc_answers_queries
    # unsqueeze to make it a 2d array
    acc_answers_queries_global_total = np.repeat(np.expand_dims(acc_answers_queries_global_total, axis=1), acc_answers_queries.shape[1], axis=1)

    # normalize aucs
    #aucs = aucs / np.max(aucs)

    plot_dict = {
        'Accuracy on Answer and Query (global)': acc_answers,
        'AUC': aucs
    }
    # plot
    plot_confidence_intervals(plot_dict, 'aucs_total_hardness.png', 'AUCs over time')