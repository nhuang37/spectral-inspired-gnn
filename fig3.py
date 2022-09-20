# -*- coding: utf-8 -*-
# @Time    : 2022/9/20 21:03
# @Author  : HCY
# @File    : fig3.py
# @Software: PyCharm

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

from power_dense_sparse_decomp import *
from dense_vs_sparse import *


def run_fig3(n, p, q, graph_seeds=30, k=2, classifier="LDA", device="cuda:0"):
    torch.manual_seed(0)

    train_pct = 0.1

    Ys = np.array([1] * (n // 2) + [-1] * (n // 2))

    results = {'ASE': [], 'Cov(X)': [], 'A_X': [], 'SGC-2': [], 'SGC-5': [], 'SGC-10': []}
    Ks = np.array([2, 5, 10])
    for i in Ks:
        results['GCN-' + str(i)] = []
        results['GCNII-' + str(i)] = []
        results['GPRGNN-' + str(i)] = []
        results['Power(Lap)-' + str(i)] = []

    for run in range(graph_seeds):
        A = construct_2BSBM(n, p, q, seed=run)
        # add self loop
        A = A + np.eye(n)
        features = gaussian_features(n, in_feats=2, cov_scale=4, seed=run)
        train_mask_all = (torch.FloatTensor(n).uniform_() > (1 - train_pct)).numpy()
        # vanilla ASE on the whole graph
        ASE_feat = emb_ASE(A, k=k)
        # cov
        Xouter = features @ features.T
        cov_feat = emb_covX(Xouter, k=k)
        # A_X
        A_X_feat = np.concatenate((ASE_feat, cov_feat), axis=1)
        # vanilla SGC on the whole graph
        SGC_1 = emb_SGC(A, features, k=k, n_layer=2, device=device)
        SGC_2 = emb_SGC(A, features, k=k, n_layer=5, device=device)
        SGC_3 = emb_SGC(A, features, k=k, n_layer=10, device=device)

        for name, emb in zip(['ASE', 'Cov(X)', 'A_X', 'SGC-2', 'SGC-5', 'SGC-10'],
                             [ASE_feat, cov_feat, A_X_feat, SGC_1, SGC_2, SGC_3]):
            if classifier == "LDA":
                acc, _, _ = feature_classify(emb, Ys, train_mask_all)
            elif classifier == "MLP":
                model = SIGN_POWER(emb.shape[1], emb.shape[1], len(np.unique(Ys)), device=device,
                                   num_hops=None, n_layers=2, dropout=0.5, input_drop=0, subnet=False).to(device)
                # = FeedForwardNet(emb.shape[1], emb.shape[1], len(torch.unique(Ys)), n_layers=2, dropout=0.1)
                acc = run_mlp(emb, model, Ys, train_mask_all, verbose=False, device=device)
                acc = float(acc)
            else:
                raise NotImplementedError
            results[name].append(acc)

        print("Finish spectral & SGC.")

        for i in Ks:
            if classifier == "MLP":
                model = GCN_Net(A, features.shape[1], Ys, n_layers=i, dropout=0.5, device=device).to(device)
                acc = run_mlp(features, model, Ys, train_mask_all, verbose=False, device=device)
                acc = float(acc)
            else:
                raise NotImplementedError
            results['GCN-' + str(i)].append(acc)

        print("Finish GCN.")

        for i in Ks:
            if classifier == "MLP":
                model = GCNII(A, features.shape[1], Ys, layer=i, dropout=0.5, device=device).to(device)
                acc = run_mlp(features, model, Ys, train_mask_all, verbose=False, device=device)
                acc = float(acc)
            else:
                raise NotImplementedError
            results['GCNII-' + str(i)].append(acc)

        print("Finish GCNII.")

        for i in Ks:
            if classifier == "MLP":
                model = GPRGNN(A, features.shape[1], Ys, K=i, dropout=0.5, device=device).to(device)
                acc = run_mlp(features, model, Ys, train_mask_all, verbose=False, device=device)
                acc = float(acc)
            else:
                raise NotImplementedError
            results['GPRGNN-' + str(i)].append(acc)

        print("Finish GPRGNN.")

        # powers
        powers = power_iterate(A, features, K=10, lap=True, include_feat=True)
        powers_tensor = torch.tensor(np.array(powers), dtype=torch.float, device=device)
        for K in Ks:
            power_emb = np.concatenate(powers[:K], axis=1)
            power_emb = torch.tensor(power_emb, dtype=torch.float, device=device)
            if classifier == "LDA":
                acc_power, _, _ = feature_classify(power_emb, Ys, train_mask_all)
            elif classifier == "MLP":
                in_size = [powers[i].shape[1] for i in range(K+1)]
                num_hidden = [input_size * 2 for input_size in in_size]
                model = SIGN_POWER(in_size, num_hidden, len(np.unique(Ys)), num_hops=K+1,
                                   n_layers=2, dropout=0.5, input_drop=0, subnet=True, device=device).to(device)
                acc_power = run_mlp(powers_tensor, model, Ys, train_mask_all, verbose=False, device=device)
                acc_power = float(acc_power)
            else:
                raise NotImplementedError
            results['Power(Lap)-' + str(K)].append(acc_power)

        print(run)

    return results


n = 500
p = 1/3
q = 1/2
graph_seeds = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# results = run_fig3(n, p, q, graph_seeds, classifier="MLP", device=device)
p_s = 1/30
q_s = 1/20
# results_sp = run_fig3(n, p_s, q_s, graph_seeds, classifier="MLP", device=device)

with open("./fig3_dense_heto.pkl", "rb") as f:
    results = pickle.load(f)

with open("./fig3_sparse_heto.pkl", "rb") as f:
    results_sp = pickle.load(f)

fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12, 4), sharey=True, sharex=True, dpi=150)

# methods = ['ASE', 'Cov(X)', 'A_X', 'SGC-2', 'SGC-5', 'SGC-10',
#            'GCN-2', 'GCN-5', 'GCN-10', 'GCNII-2', 'GCNII-5', 'GCNII-10', 'GPRGNN-2', 'GPRGNN-5', 'GPRGNN-10',
#            'Power(Lap)-2', 'Power(Lap)-5', 'Power(Lap)-10']

methods = ['ASE', 'Cov(X)', 'A_X',
           'GCN-2', 'GCN-5', 'GCN-10',
           'Power(Lap)-2', 'Power(Lap)-5', 'Power(Lap)-10']

results_new = {}
results_sp_new = {}
for m in methods:
    results_new[m] = results[m]
    results_sp_new[m] = results_sp[m]

df = pd.DataFrame.from_dict(results_new)
pd.DataFrame(df).boxplot(grid=False, rot=18, fontsize=12, ax=axs[0])
axs[0].set_ylabel('Accuracy', fontsize=12)
axs[0].set_title(f"p={p:.2f}, q={q:.2f}", fontsize=15)

df = pd.DataFrame.from_dict(results_sp_new)
pd.DataFrame(df).boxplot(grid=False, rot=18, fontsize=12, ax=axs[1])
# axs[1].set_ylabel('Accuracy', fontsize=12)
axs[1].set_title(f"p={p_s:.2f}, q={q_s:.2f}", fontsize=15)

fig.tight_layout()
plt.show()
