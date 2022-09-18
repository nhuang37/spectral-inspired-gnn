# -*- coding: utf-8 -*-
# @Time    : 2022/8/26 11:37
# @Author  : HCY
# @File    : dense_vs_sparse.py
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
from dgl.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
import time
from PT import PropTrain, train, test, label_propagation
from early_stop import EarlyStopping, Stop_args

from power_dense_sparse_decomp import get_angle, compute_pairs, power_iterate, construct_2BSBM, construct_graph, \
    gaussian_features, feature_classify, emb_ASE, emb_covX, emb_SGC, kcore_classify
# from power import SIGN_POWER, FeedForwardNet


#  ##inception
class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN_POWER(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_hops, n_layers, dropout, input_drop, subnet=True, device="cuda:0"):
        '''
        in_feats, hidden: lists of input features/hidden dimension
        '''
        super(SIGN_POWER, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.input_drop = nn.Dropout(input_drop)
        self.subnet = subnet
        if self.subnet:  # MLP subnets for each power features
            self.inception_ffs = nn.ModuleList()
            for hop in range(num_hops):
                self.inception_ffs.append(FeedForwardNet(in_feats[hop], hidden[hop], hidden[hop], n_layers, dropout))
            hidden_sum = np.array(hidden).sum()
            self.project = FeedForwardNet(hidden_sum, hidden_sum, out_feats, n_layers, dropout)
        else:  # MLP for last iterate features
            self.project = FeedForwardNet(in_feats, hidden, out_feats, n_layers, dropout)

    def forward(self, feats):
        if self.subnet:
            feats = [self.input_drop(feat.to(self.device)) for feat in feats]
            feats = torch.stack(feats)
            # feats.to(self.device)
            hidden = []
            # concatenate outputs from each subnets (size n by d*num_hops), followed by a MLP
            for i, (feat, ff) in enumerate(zip(feats, self.inception_ffs)):
                hidden.append(ff(feat))
            out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        else:  # one linear output layer
            # feats = torch.cat(feats, dim=-1)
            feats = torch.tensor(feats, dtype=torch.float, device=self.device)
            out = self.project(feats)
        return out

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()


class GCNII(nn.Module):
    def __init__(self, A, num_feat, Ys, layer=1, alpha=0.1, device="cuda:0"):
        super(GCNII, self).__init__()
        edges = np.where(A == 1)
        g = dgl.graph(edges, device=device)
        self.device = device
        self.g = g
        self.num_feat = num_feat
        self.Ys = Ys
        self.layer = layer
        self.alpha = alpha
        self.embedding = GCN2Conv(num_feat, layer=layer, alpha=alpha)  # , activation=F.relu)
        self.classifier = SIGN_POWER(num_feat, num_feat, len(np.unique(Ys)),
                                     num_hops=None, n_layers=2, dropout=0.1, input_drop=0, subnet=False)

    def forward(self, feats):
        feats = torch.tensor(feats, dtype=torch.float, device=self.device)
        emb = feats
        emb = self.embedding(self.g, emb, feats)
        out = self.classifier(emb)
        return out


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, device="cuda:0", Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.device = device

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP, device=device))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, A, num_feat, Ys, K=10, alpha=0.1,
                 Init='Random', Gamma=None, ppnp='GPR_prop', dprate=0.5, device="cuda:0"):
        super(GPRGNN, self).__init__()
        self.classifier = SIGN_POWER(num_feat, num_feat, len(np.unique(Ys)),
                                     num_hops=None, n_layers=2, dropout=0.1, input_drop=0, subnet=False)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.A = A
        self.Ys = Ys
        self.num_feat = num_feat
        self.device = device

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x):
        edge_index = torch.stack(torch.where(torch.tensor(self.A, dtype=torch.float, device=self.device)))

        x = self.classifier(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)


class GCN_Net(torch.nn.Module):
    def __init__(self, A, num_feat, Ys, dropout=0.1, n_layers=2, device="cuda:0"):
        super(GCN_Net, self).__init__()
        self.layers = nn.ModuleList()
        self.A = A
        self.Ys = Ys
        self.num_feat = num_feat
        self.device = device
        self.n_layers = n_layers

        if n_layers == 1:
            self.layers.append(GCNConv(num_feat, len(np.unique(Ys))))
        else:
            self.layers.append(GCNConv(num_feat, num_feat * 2))
            for i in range(n_layers - 2):
                self.layers.append(GCNConv(num_feat * 2, num_feat * 2))
            self.layers.append(GCNConv(num_feat * 2, len(np.unique(Ys))))

        if n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        edge_index = torch.stack(torch.where(torch.tensor(self.A, dtype=torch.float, device=self.device)))
        x = torch.tensor(x, dtype=torch.float, device=self.device)
        for layer_id, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return F.log_softmax(x, dim=1)


class Semi_POWER(torch.nn.Module):
    def __init__(self, g, num_feat, Ys, K, rw=False, lap=False, device="cuda:0"):
        """
            Precompute power-iterated features
            g: graph, numpy array (n by n)
            feat: features
            If rw: use D^{-1}A instead of A as the graph operators
        """
        super(Semi_POWER, self).__init__()
        self.g = g
        self.K = K
        self.rw = rw
        self.lap = lap
        self.device = device
        self.layers = nn.ModuleList()
        for i in range(K):
            self.layers.append(nn.Linear(num_feat, num_feat))
        in_size = [num_feat] * (K + 1)
        num_hidden = [input_size * 2 for input_size in in_size]
        self.classifer = SIGN_POWER(in_size, num_hidden, len(np.unique(Ys)), num_hops=K+1,
                                    n_layers=2, dropout=0.1, input_drop=0, subnet=True, device=device)

    def forward(self, feat):
        feat = torch.tensor(feat, dtype=torch.float, device=self.device)
        g = torch.tensor(self.g, dtype=torch.float, device=self.device)
        # g = self.g
        powers = [feat]
        if self.rw:
            # Dinv = 1 / self.g.sum(axis=1)
            Dinv = 1 / torch.sum(g, dim=1)
            g = g * Dinv[:, None]  # (500,) --> (500, 1)
        for iter in range(self.K):
            # message passing
            tildeU = g @ feat
            tildeU = self.layers[iter](tildeU)  # semi-supervised
            # normalization
            normalizer = torch.linalg.pinv(tildeU.T @ tildeU)
            feat = tildeU @ normalizer  # scaled eigenvector (arbitrary rotated if upper_tri is False)
            col_norm = torch.linalg.norm(feat, axis=0)
            powers.append(feat / col_norm)
        powers = torch.stack(powers)
        out = self.classifer(powers)
        return out


class Semi_SGC(torch.nn.Module):
    def __init__(self, A, num_feat, Ys, K, device="cuda:0"):
        self.A = A
        self.K = K
        self.num_feat = num_feat
        self.Ys = Ys
        self.device = device
        self.W = nn.Linear(num_feat, num_feat * 2)
        self.classifier = SIGN_POWER(num_feat * 2, num_feat * 2, len(np.unique(Ys)),
                                     num_hops=None, n_layers=2, dropout=0.1, input_drop=0, subnet=False)

    def forward(self, feat):
        feat = torch.tensor(feat, dtype=torch.float, device=self.device)
        A = torch.tensor(self.A, dtype=torch.float, device=self.device)
        Dinv = 1 / A.sum(axis=1)
        A_rw = A * Dinv[:, None]
        emb = feat
        for layer in range(self.K):
            emb = A_rw @ emb
        emb = self.W(emb)
        out = self.classifier(emb)
        return out


def train(emb, model, Ys, train_mask, loss_fcn, optimizer, device):
    model.train()
    Ys[Ys == -1] = 0
    Ys = torch.tensor(Ys, dtype=torch.long, device=device)
    loss = loss_fcn(model(emb)[train_mask], Ys[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test(emb, model, Ys, train_mask, device):
    model.eval()
    Ys[Ys == -1] = 0
    Ys = torch.tensor(Ys, dtype=torch.long, device=device)
    preds = torch.argmax(model(emb)[~train_mask], dim=-1)
    acc = (preds == Ys[~train_mask]).sum() / len(Ys[~train_mask])
    return acc


def run_mlp(emb, model, Ys, train_mask, lr=1e-2, weight_decay=0,
            num_epochs=100, eval_every=1, verbose=True, device="cuda:0"):
    # optim
    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Start training
    best_test = 0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train(emb, model, Ys, train_mask, loss_fcn, optimizer, device)

        if epoch % eval_every == 0:
            with torch.no_grad():
                acc = test(emb, model, Ys, train_mask, device)
            end = time.time()
            log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
            log += "Acc: Test {:.4f}".format(acc)
            if verbose:
                print(log)
            if acc > best_test:
                best_test = acc
                best_epoch = epoch
    if verbose:
        print("Best Epoch {}, Test {:.4f}".format(best_epoch, best_test))

    return best_test


def run_simulation(n, p, q, graph_seeds=30, k=2, classifier="LDA", device="cuda:0"):
    torch.manual_seed(0)

    train_pct = 0.1

    Ys = np.array([1] * (n // 2) + [-1] * (n // 2))
    results = {}
    # results = {'ASE': [], 'Cov(X)': [], 'A_X': [], 'MP-2': [], 'MP-5': [], 'MP-10': []}#,
    #            # 'decomp(ASE)': [], 'decomp(Power)': []}
    Ks = np.array([2, 5, 10])
    for K in Ks:
        results['Power_' + str(K)] = []
        # results['Semi-Power_' + str(K)] = []
    #
    # GCNII_layer = np.array([4, 32, 64])
    # for i in GCNII_layer:
    #     results['GCNII_' + str(i)] = []

    # GPRGNN_layer = np.array([2, 5, 10])
    # for i in GPRGNN_layer:
    #     results['GPRGNN_' + str(i)] = []

    # GCN_layer = np.array([2, 5, 10])
    # for i in GCN_layer:
    #     results['GCN_' + str(i)] = []

    for run in range(graph_seeds):
        A = construct_2BSBM(n, p, q, seed=run)
        # add self loop
        A = A + np.eye(n)
        features = gaussian_features(n, in_feats=2, cov_scale=4, seed=run)
        train_mask_all = (torch.FloatTensor(n).uniform_() > (1 - train_pct)).numpy()
        # # vanilla ASE on the whole graph
        # ASE_feat = emb_ASE(A, k=k)
        # # cov
        # Xouter = features @ features.T
        # cov_feat = emb_covX(Xouter, k=k)
        # # A_X
        # A_X_feat = np.concatenate((ASE_feat, cov_feat), axis=1)
        # # vanilla SGC on the whole graph
        # SGC_1 = emb_SGC(A, features, k=k, n_layer=2)
        # SGC_2 = emb_SGC(A, features, k=k, n_layer=5)
        # SGC_3 = emb_SGC(A, features, k=k, n_layer=10)
        #
        # for name, emb in zip(['ASE', 'Cov(X)', 'A_X', 'MP-2', 'MP-5', 'MP-10'],
        #                      [ASE_feat, cov_feat, A_X_feat, SGC_1, SGC_2, SGC_3]):
        #     if classifier == "LDA":
        #         acc, _, _ = feature_classify(emb, Ys, train_mask_all)
        #     elif classifier == "MLP":
        #         model = SIGN_POWER(emb.shape[1], emb.shape[1], len(np.unique(Ys)), device=device,
        #                            num_hops=None, n_layers=2, dropout=0.1, input_drop=0, subnet=False).to(device)
        #         # = FeedForwardNet(emb.shape[1], emb.shape[1], len(torch.unique(Ys)), n_layers=2, dropout=0.1)
        #         acc = run_mlp(emb, model, Ys, train_mask_all, verbose=False, device=device)
        #         acc = float(acc)
        #     else:
        #         raise NotImplementedError
        #     results[name].append(acc)

        # print("Finish spectral.")

        # for i in GCNII_layer:
        #     if classifier == "LDA":
        #         emb = emb_GCNII(A, features, i, 0.1)
        #         acc, _, _ = feature_classify(emb, Ys, train_mask_all)
        #     elif classifier == "MLP":
        #         model = GCNII(A, features.shape[1], Ys, layer=i, device=device).to(device)
        #         acc = run_mlp(features, model, Ys, train_mask_all, verbose=False, device=device)
        #         acc = float(acc)
        #     else:
        #         raise NotImplementedError
        #     results['GCNII_' + str(i)].append(acc)

        # for i in GPRGNN_layer:
        #     if classifier == "MLP":
        #         model = GPRGNN(A, features.shape[1], Ys, K=i, device=device).to(device)
        #         acc = run_mlp(features, model, Ys, train_mask_all, verbose=False, device=device)
        #         acc = float(acc)
        #     else:
        #         raise NotImplementedError
        #     results['GPRGNN_' + str(i)].append(acc)

        # for i in GCN_layer:
        #     if classifier == "MLP":
        #         model = GCN_Net(A, features.shape[1], Ys, n_layers=i, device=device).to(device)
        #         acc = run_mlp(features, model, Ys, train_mask_all, verbose=False, device=device)
        #         acc = float(acc)
        #     else:
        #         raise NotImplementedError
        #     results['GCN_' + str(i)].append(acc)

        # print("Finish GCNII.")

        # powers
        powers = power_iterate(A, features, K=10, include_feat=True)
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
                                   n_layers=2, dropout=0.1, input_drop=0, subnet=True, device=device).to(device)
                acc_power = run_mlp(powers_tensor, model, Ys, train_mask_all, verbose=False, device=device)
                acc_power = float(acc_power)
            else:
                raise NotImplementedError
            results['Power_' + str(K)].append(acc_power)

        # semi-power
        # for K in Ks:
        #     if classifier == "MLP":
        #         model = Semi_POWER(A, features.shape[1], Ys, K, device=device).to(device)
        #         acc = run_mlp(features, model, Ys, train_mask_all, verbose=False, device=device)
        #         acc = float(acc)
        #     else:
        #         raise NotImplementedError
        #     results['Semi-Power_' + str(K)].append(acc)

        print(run)

    return results


if __name__ == "__main__":
    # teresa's simulation: test subspace iteration for GNNs - dense
    n = 500
    p = 1 / 2
    q = 1 / 3
    graph_seeds = 30
    divide_num = 15
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(divide_num):
        results.append(run_simulation(n, p/(i+1), q/(i+1), graph_seeds, classifier="MLP", device=device))
    with open("results_Power_A_includefeat.pkl", "wb") as tf:
        pickle.dump(results, tf)

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 6), sharey=True, sharex=True, dpi=150)
    for index, name in enumerate(list(results[0].keys())):
        row, col = int(index / 3), index % 3
        dict = {}
        for i in range(divide_num):
            dict["1/%d"%(i+1)] = results[i][name]
        df = pd.DataFrame.from_dict(dict)
        df.boxplot(grid=False, rot=30, fontsize=12, ax=axs[row, col])
        axs[row, col].set_ylabel('Accuracy', fontsize=12)
        axs[row, col].set_title(name, fontsize=15)  # n={n}, train_pct=0.1, MC_runs={graph_seeds}
    fig.tight_layout()
    plt.show()

    # n = 500
    # p = 1 / 2
    # q = 1 / 3
    # graph_seeds = 30
    # results = run_simulation(n, p, q, graph_seeds)
    # p_s = 1 / 20
    # q_s = 1 / 30
    # results_sp = run_simulation(n, p_s, q_s, graph_seeds)
    #
    # plt.figure(figsize=(10, 4), dpi=100)
    # df = pd.DataFrame.from_dict(results)
    # df.boxplot(grid=False, rot=18, fontsize=12)
    # plt.ylabel('Accuracy', fontsize=12)
    # plt.title(f"p={p:.2f}, q={q:.2f}", fontsize=15)  # n={n}, train_pct=0.1, MC_runs={graph_seeds}
    # plt.show()
    #
    # plt.figure(figsize=(10, 4), dpi=100)
    # df = pd.DataFrame.from_dict(results)
    # df.boxplot(grid=False, rot=18, fontsize=12)
    # plt.ylabel('Accuracy', fontsize=12)
    # plt.title(f"p={p:.2f}, q={q:.2f}", fontsize=15)  # n={n}, train_pct=0.1, MC_runs={graph_seeds}
    # plt.show()
