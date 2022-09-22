import math
import random
import os
import numpy as np
import pandas as pd
from numpy import linalg as LA
import numpy.random as npr
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
from scipy.sparse.linalg import eigsh, eigs
from scipy.linalg import hadamard, subspace_angles
import itertools
import dgl
# from dgl.nn import GCN2Conv

from utils import run_fig3


if __name__ == "__main__":
    n = 500
    p = 1 / 2
    q = 1 / 3
    graph_seeds = 30
    divide_num = 15
    results = []
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    for i in range(divide_num):
        results.append(run_fig3(n, p / (i + 1), q / (i + 1), graph_seeds, classifier="MLP", device=device))
    # with open("results.pkl", "wb") as tf:
    #     pickle.dump(results, tf)

    names = ['ASE', 'Cov(X)', 'A_X',
             'Power-last-iter(Lap)-2', 'Power-last-iter(Lap)-5', 'Power-last-iter(Lap)-10',
             'SGC-2', 'SGC-5', 'SGC-10',
             'GCN-2', 'GCN-5', 'GCN-10',
             'GCNII-2', 'GCNII-5', 'GCNII-10',
             'GPR-GNN-2', 'GPR-GNN-5', 'GPR-GNN-10',
             'SIGN-2', 'SIGN-5', 'SIGN-10',
             'Power(Lap)-2', 'Power(Lap)-5', 'Power(Lap)-10']

    fig, axs = plt.subplots(ncols=3, nrows=8, figsize=(15, 18), sharey=True, sharex=True, dpi=150)

    for index, name in enumerate(names):
        # for index, name in enumerate(names):
        row, col = int(index / 3), index % 3
        dict = {}
        for i in range(divide_num):
            dict["1/%d" % (i + 1)] = results[i][name]

        Q1 = []
        Q2 = []
        Q3 = []
        for key in dict.keys():
            Q1.append(np.quantile(dict[key], 0.25))
            Q2.append(np.quantile(dict[key], 0.5))
            Q3.append(np.quantile(dict[key], 0.75))
        if col == 0:
            axs[row, col].plot(list(dict.keys()), Q2, color='purple', lw=0.5, ls='-', marker='o', ms=4)
            axs[row, col].fill_between(list(dict.keys()), Q3, Q1, color=(229 / 256, 204 / 256, 249 / 256), alpha=0.9)
        elif col == 1:
            axs[row, col].plot(list(dict.keys()), Q2, color='green', lw=0.5, marker='^', ms=4)
            axs[row, col].fill_between(list(dict.keys()), Q3, Q1, color=(204 / 256, 236 / 256, 223 / 256), alpha=0.9)
        else:
            axs[row, col].plot(list(dict.keys()), Q2, color='blue', lw=0.5, ls='-.', marker='s', ms=4)
            axs[row, col].fill_between(list(dict.keys()), Q3, Q1, color=(191 / 256, 191 / 256, 255 / 256), alpha=0.9)
        if col == 0:
            axs[row, col].set_ylabel('Accuracy', fontsize=12)
        if row == 7:
            axs[row, col].set_xlabel('Density', fontsize=12)
        axs[row, col].set_xticklabels(list(dict.keys()), rotation=30)
        axs[row, col].set_title(name, fontsize=15)
    fig.tight_layout()
    plt.show()

