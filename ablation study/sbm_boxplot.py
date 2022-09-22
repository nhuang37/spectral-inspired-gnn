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
    graph_seeds = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = ["./fig3_dense.pkl", "./fig3_sparse.pkl", "./fig3_dense_heto.pkl", "./fig3_sparse_heto.pkl"]
    p = [1/2, 1/20, 1/3, 1/30]
    q = [1/3, 1/30, 1/2, 1/20]
    num = ["(a)", "(b)", "(c)", "(d)"]

    for i in range(len(path)):
        results = run_fig3(n, p[i], q[i], graph_seeds, classifier="MLP", device=device)
        with open(path[i], "wb") as f:
            pickle.dump(results, f)

    methods = ['ASE', 'Cov(X)', 'A_X', 'SGC-2', 'SGC-5', 'SGC-10',
               'GCN-2', 'GCN-5', 'GCN-10', 'GCNII-2', 'GCNII-5', 'GCNII-10', 'GPRGNN-2', 'GPRGNN-5', 'GPRGNN-10',
               'Power-last-iter(Lap)-2', 'Power-last-iter(Lap)-5', 'Power-last-iter(Lap)-10']

    model = ['Spectral', 'SGC', 'GCN', 'GCNII', 'GPR-GNN', 'Power-last-iter(Lap)']
    layers = [2, 5, 10]

    fig, axs = plt.subplots(ncols=len(model), nrows=4, figsize=(11, 10), sharey=True, sharex=False, dpi=150)

    for row in range(4):
        with open(path[row], "rb") as f:
            results = pickle.load(f)

        for col in range(len(model)):
            sub = {}
            if col == 0:
                for j in range(3):
                    sub[methods[col * 3 + j]] = results[methods[col * 3 + j]]
                axs[row, col].set_ylabel(f"p={p[row]:.2f}, q={q[row]:.2f} " + num[row], fontsize=12, fontweight='bold')
            else:
                for j in range(3):
                    sub[layers[j]] = results[methods[col * 3 + j]]
            df = pd.DataFrame.from_dict(sub)
            pd.DataFrame(df).boxplot(grid=False, rot=18, fontsize=10, ax=axs[row, col])
            if row != 3:
                axs[row, col].axes.xaxis.set_ticklabels([])
            if row == 3:
                axs[row, col].set_title(model[col], fontsize=13, y=-0.3)

    # fig.suptitle("(A) Homophilous Graphs", fontsize=16)
    fig.tight_layout()
    plt.show()
