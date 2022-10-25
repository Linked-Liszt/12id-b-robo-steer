import utils.data as du
import random
from matplotlib import pyplot as plt
import torch
import numpy as np
import dataclasses

@dataclasses.dataclass
class HistInfo():
    bin_sizes: np.ndarray
    bins: np.ndarray

def graph_sample(model, dataset: du.PosDataDataset, ds_subset):
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


def graph_ers(model, dataset: du.PosDataDataset, ds_subset, high=True):
    fig, axs = plt.subplots(5, 5)
    fig.set_size_inches(10, 5)
    idxs = list(range(len(ds_subset.indices)))
    errs = []
    preds = []
    for idx in idxs:
        scan = dataset.scans[ds_subset.indices[idxs[idx]]]
        with torch.no_grad():
            model_pred = model(torch.Tensor(scan.reads)).item()
        errs.append(np.abs(model_pred - scan.usr_pos))
        preds.append(model_pred)

    err_idxs = np.argsort(errs)
    if high:
        err_idxs = np.flip(err_idxs)

    i = 0
    for row in axs:
        for ax in row:
            scan = dataset.scans[ds_subset.indices[err_idxs[i]]]
            ax.plot(scan.pos, scan.reads)
            ax.plot(scan.usr_pos, scan.usr_read, marker='o')
            pred_point = np.interp([preds[err_idxs[i]]], scan.pos, scan.reads)
            ax.plot(preds[err_idxs[i]], pred_point, marker='x')
            i += 1
            
    return fig


def plot_diff_hist(model, dataset: du.PosDataDataset, ds_subset):
    fig, ax = plt.subplots()

    diffs = []
    for i in ds_subset.indices:
        scan = dataset.scans[i]
        with torch.no_grad():
            model_pred = model(torch.Tensor(scan.reads)).item()
        diffs.append(np.abs(model_pred - scan.usr_pos))

    bin_sizes, bins, _ = ax.hist(diffs, bins=20, range=(0.0, 2.0))
    return fig, HistInfo(bin_sizes, bins)
    

def print_hist_info(hist_info: HistInfo) -> None:
    print(f'Total Range: {hist_info.bins[0]:.2f}-{hist_info.bins[-1]:.2f}')
    print(f'Err Range: [range_low]-[range_high]: [num_samples] [% of total samples]')
    total_count = round(np.sum(hist_info.bin_sizes))
    for i in range(len(hist_info.bin_sizes)):
        if round(hist_info.bin_sizes[i]) > 0:
            print(f'Err Range {hist_info.bins[i]:.2f}-{hist_info.bins[i+1]:.2f}: '
                  + f'{round(hist_info.bin_sizes[i])} {hist_info.bin_sizes[i] / total_count * 100:.2f}%')


def plot_scans(scan_list):
    fig, ax = plt.subplots(len(scan_list))
    for ax, scan in zip(ax, scan_list):
        ax.plot(scan.pos, scan.reads)
        ax.plot(scan.usr_pos, scan.usr_read, marker='o')
    return fig