import utils.data as du
import random
from matplotlib import pyplot as plt
import torch
import numpy as np

def graph_sample(model, dataset: du.PosDataDatset, ds_subset):
    fig, axs = plt.subplots(5, 5)
    fig.set_size_inches(10, 5)
    idxs = list(range(len(ds_subset.indices)))
    random.shuffle(idxs)
    i = 0
    for row in axs:
        for ax in row:
            scan = dataset.scans[ds_subset.indices[idxs[i]]]
            i += 1
            ax.plot(scan.pos, scan.reads)
            ax.plot(scan.usr_pos, scan.usr_read, marker='o')
            with torch.no_grad():
                model_pred = model(torch.Tensor(scan.reads)).item()
            pred_point = np.interp([model_pred], scan.pos, scan.reads)
            ax.plot(model_pred, pred_point, marker='x')
    return fig

def plot_diff_hist(model, dataset: du.PosDataDatset, ds_subset):
    fig, ax = plt.subplots()

    diffs = []
    for i in ds_subset.indices:
        scan = dataset.scans[i]
        with torch.no_grad():
            model_pred = model(torch.Tensor(scan.reads)).item()
        diffs.append(np.abs(model_pred - scan.usr_pos))

    ax.hist(diffs, bins=20, range=(0.0, 2.0))
    return fig
    