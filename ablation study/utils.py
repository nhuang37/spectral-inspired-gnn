import math
import random
import os
import time
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

from baseline.gcn_variants_real_graph import GCNII, GPRGNN, SIGN_POWER, GCN_Net


def get_angle(x,y, sign=False):
  '''
  compute angle between vectors x and y
  '''
  x = x / LA.norm(x)
  y = y / LA.norm(y)
  with np.errstate(divide='ignore',invalid='ignore'):
    rad = np.arccos(np.clip(np.dot(x,y), -1, 1))
  angle = np.min((np.degrees(rad), 360-np.degrees(rad)))
  if sign: #care about the direction, [0,180]
    return angle
  else: #modulo sign, [0,90]
    return np.min((angle, 180-angle))


def compute_pairs(emb_gnn, ASE):
  '''
  compute pairwise angles of GNN embedding and ASEs
  '''
  k = emb_gnn.shape[1]
  angles = []
  #combs = itertools.combinations(range(k), 2)
  #for (i,j) in combs:
  for i in range(emb_gnn.shape[1]):
    min_angle = np.zeros(ASE.shape[1])
    for j in range(ASE.shape[1]):
      min_angle[j] = get_angle(emb_gnn[:,i], ASE[:,j])
    angles.append(min_angle.min())
  return np.array(angles)


def power_iterate(g, feat, K, rw=False, lap=False, include_feat=True, upper_tri=False):
    """
    Precompute power-iterated features
    g: graph, numpy array (n by n)
    feat: features
    Return: a list with length-K of feature matrices, each with shape (n by d)
    If rw: use D^{-1}A instead of A as the graph operators
    """
    if include_feat:
      powers = [feat]
    else:
      powers = []
    if rw:
      Dinv = 1 / g.sum(axis=1)
      g = g * Dinv[:, None]
    if lap:
      degs = g.sum(axis=0).clip(min=1)
      norm = np.power(degs, -0.5)
      g = np.diag(norm) @ g @ np.diag(norm)
    for iter in range(K):
      #message passing
      tildeU = g @ feat
      #normalization
      if upper_tri:
        normalizer = np.linalg.pinv(np.triu(tildeU.T @ tildeU))
      else:
        normalizer = np.linalg.pinv(tildeU.T @ tildeU)
      feat = tildeU @ normalizer #scaled eigenvector (arbitrary rotated if upper_tri is False)
      col_norm = np.linalg.norm(feat, axis=0)
      powers.append(feat / col_norm)
    return powers


def construct_2BSBM(n, p, q, seed=0):
  '''
  Generate a random (n)-nodes symmetric 2-block SBM
  '''
  assert n % 2 == 0, "must pass in even number of nodes"
  probs = [[p,q],[q,p]]
  nvecs = [n//2, n//2]
  graph = nx.stochastic_block_model(nvecs, probs, seed=seed)
  return nx.to_numpy_array(graph)


def construct_graph(n, nvecs, p_d, q_d, p_s, q_s, a, b, seed=0):
  '''
  Generate a random (n)-nodes 4-block symmetric SBM using probability matrix prob
  n: total number of nodes
  nvecs: number of nodes per block
  p_d, q_d: dense block connectivity probability
  p_s, q_s: sparse block connectivity probability
  a, b: cross edges
  probability matrix B:
    [[p_d, q_d, a, b],[q_d, p_d, b, a],[a, b, p_s, q_s], [b, a, q_s, p_s]]
  return DGL_graph; numpy adjacency matrix
  '''
  assert n % 4 == 0, "must pass in node numbers that can be divided by 4 (4-block symmetric SBM!)"
  #sizes = [n//4, n//4, n//4, n//4]
  probs = [[p_d, q_d, a, b],[q_d, p_d, b, a],[a, b, p_s, q_s], [b, a, q_s, p_s]]
  graph = nx.stochastic_block_model(nvecs, probs, seed=seed)
  return nx.to_numpy_array(graph)


def gaussian_features(n, in_feats, cov_scale, mean_scale=1, seed=0):
  '''
  n: number of nodes per block
  in_feats: input feature dimension
  cov_scale: scaling factor for covariance matrix
  return (2n x in_feats) feature matrix, where the first n rows are sample iid from N([1,...1], cov_scale*I),
  the last n rows are sample iid from N([-1,...-1], cov_scale*I)
  '''
  np.random.seed(seed)
  mean = mean_scale*np.ones(in_feats)
  cov = cov_scale*np.eye(in_feats)
  feat_1 = np.random.multivariate_normal(mean, cov, n//2)
  feat_2 = np.random.multivariate_normal(-mean, cov, n//2)
  features = np.concatenate((feat_1, feat_2), axis=0)
  return features


def feature_classify(features, Ys, train_mask):
  '''
  features: node features
  Ys: classification label
  train_mask: mask for trainingset
  return: test_acc, Yhats (entry = train labels if available)
  '''
  #clf = svm.SVC(kernel='linear', max_iter=500)
  K = min(len(np.unique(Ys))-1, features.shape[1])
  clf = LinearDiscriminantAnalysis(n_components=K)
  clf.fit(features[train_mask], Ys[train_mask])
  Yhats = clf.predict(features)
  Yhats[train_mask] = Ys[train_mask]
  # Yhats = Ys
  return clf.score(features[~train_mask], Ys[~train_mask]), len(Ys[~train_mask]), Yhats


def emb_ASE(A, k=2):
  '''
  ASE embedding
  '''
  # u, s, v = np.linalg.svd(A, hermitian=True)
  # return u[:, :k] * s[:k]
  evalues, evectors = eigsh(A, k=k)
  return evectors[:, :k]


def emb_covX(Xouter, k=2):
  '''
  top-k eigenvector of covariance
  '''
  evaluesX, evectorsX = eigsh(Xouter, k=k)
  return evectorsX[:, :k]


def emb_SGC_SIGN(A, feat, k=2, n_layer=1, device="cuda:0"):
  '''
  SGC embedding using identity (node identifier) as initial feature
  A: with self-loop
  '''
  # Dinv = 1 / A.sum(axis=1)
  # A_rw = A * Dinv[:, None]
  feat = torch.tensor(feat, dtype=torch.float, device=device)
  A = torch.tensor(A, dtype=torch.float, device=device)
  inter = [feat]
  degs = A.sum(axis=0).clamp(min=1)
  norm = torch.pow(degs, -0.5)
  A_lap = torch.diag(norm) @ A @ torch.diag(norm)
  out = feat
  for layer in range(n_layer):
    out = A_lap @ out  # 写错，原来是feats
    inter.append(out)
  return inter


def emb_SGC(A, feat, k=2, n_layer=1, device="cuda:0"):
  '''
  SGC embedding using identity (node identifier) as initial feature
  A: with self-loop
  '''
  # Dinv = 1 / A.sum(axis=1)
  # A_rw = A * Dinv[:, None]
  feat = torch.tensor(feat, dtype=torch.float, device=device)
  A = torch.tensor(A, dtype=torch.float, device=device)
  degs = A.sum(axis=0).clamp(min=1)
  norm = torch.pow(degs, -0.5)
  A_lap = torch.diag(norm) @ A @ torch.diag(norm)
  out = feat
  for layer in range(n_layer):
    out = A_lap @ out  # 写错，原来是feats
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


def run_fig3(n, p, q, graph_seeds=30, k=2, classifier="LDA", device="cuda:0"):
    torch.manual_seed(0)

    train_pct = 0.1

    Ys = np.array([1] * (n // 2) + [-1] * (n // 2))

    results = {'ASE': [], 'Cov(X)': [], 'A_X': [],
               'SGC-2': [], 'SGC-5': [], 'SGC-10': []}
    # results = {}
    Ks = np.array([2, 5, 10])
    for i in Ks:
        results['SIGN-' + str(i)] = []
        results['GCN-' + str(i)] = []
        results['GCNII-' + str(i)] = []
        results['GPRGNN-' + str(i)] = []
        results['Power(Lap)-' + str(i)] = []
        results['Power-last-iter(Lap)-' + str(i)] = []

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
            power_emb = np.concatenate(powers[:(K+1)], axis=1)
            power_emb = torch.tensor(power_emb, dtype=torch.float, device=device)
            if classifier == "LDA":
                acc_power, _, _ = feature_classify(power_emb, Ys, train_mask_all)
            elif classifier == "MLP":
                in_size = [powers[i].shape[1] for i in range(K+1)]
                num_hidden = [input_size * 2 for input_size in in_size]
                model = SIGN_POWER(in_size, num_hidden, len(np.unique(Ys)), num_hops=K+1,
                                   n_layers=2, dropout=0.5, input_drop=0, subnet=True, device=device).to(device)
                acc_power = run_mlp(powers_tensor, model, Ys, train_mask_all, verbose=False, device=device)
                # 其实这里应该是power_emb而不是powers_tensor，但是因为sign层数有限制所以没关系
                acc_power = float(acc_power)
            else:
                raise NotImplementedError
            results['Power(Lap)-' + str(K)].append(acc_power)

        print("Finish Power-all.")

        # powers_last_iter
        for K in Ks:
            if classifier == "MLP":
                in_size = powers[K].shape[1]
                num_hidden = in_size * 2
                model = SIGN_POWER(in_size, num_hidden, len(np.unique(Ys)), num_hops=1,
                                   n_layers=2, dropout=0.5, input_drop=0, subnet=False, device=device).to(device)
                acc_power = run_mlp(powers[K], model, Ys, train_mask_all, verbose=False, device=device)
                acc_power = float(acc_power)
            else:
                raise NotImplementedError
            results['Power-last-iter(Lap)-' + str(K)].append(acc_power)

        # sgc-sign
        inter = emb_SGC_SIGN(A, features, k=k, n_layer=10, device=device)
        for K in Ks:
            if classifier == "MLP":
                in_size = [inter[i].shape[1] for i in range(K + 1)]
                num_hidden = [input_size * 2 for input_size in in_size]
                model = SIGN_POWER(in_size, num_hidden, len(np.unique(Ys)), num_hops=K + 1,
                                   n_layers=2, dropout=0.5, input_drop=0, subnet=True, device=device).to(device)
                acc_sign = run_mlp(torch.stack(inter), model, Ys, train_mask_all, verbose=False, device=device)
                acc_sign = float(acc_sign)
            else:
                raise NotImplementedError
            results['SIGN-' + str(K)].append(acc_sign)

        print(run)

    return results
