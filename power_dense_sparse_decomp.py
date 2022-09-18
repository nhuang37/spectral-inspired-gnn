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
from PT import PropTrain, train, test, label_propagation
from early_stop import EarlyStopping, Stop_args

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

def power_iterate(g, feat, K, rw=False, include_feat=True, upper_tri=False):
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

def emb_SGC(A, feats, k=2, n_layer=1):
  '''
  SGC embedding using identity (node identifier) as initial feature
  A: with self-loop
  '''
  Dinv = 1 / A.sum(axis=1)
  A_rw = A * Dinv[:, None]
  out = feats
  for layer in range(n_layer):
    out = A_rw @ out  # 写错，原来是feats
  return out


# def emb_GCNII(A, feats, n_layer=1, alpha=0.1):
#     feats = torch.tensor(feats, dtype=torch.float)
#     edges = np.where(A == 1)
#     g = dgl.graph(edges)
#     num_feat = feats.shape[1]
#     conv = GCN2Conv(num_feat, layer=n_layer, alpha=alpha)  # , activation=F.relu)
#     res = feats
#     res = conv(g, res, feats)
#     return res.detach().numpy()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def kcore_classify(A, nvecs, Ys, train_mask, oracle=True, kc=None, k=2, prop_steps=3, verbose=False, power=False, feat=None,
                   NI=False, ks=False, degree=False, LP=False, PT=False, leader=False, mode=0):
  '''
  Graph = dense + sparse
  1. Obtain kcore (dense) part of the graph, run emb_ASE + classify => obtain Yhats for dense nodes
  2. Perform majority vote (type) label propogation for sparse node => obtain Yhats for sparse ndoes
  - 2.1 Use majority mean from dense nodes as features for sparse nodes + classify
  '''
  n = A.shape[0]
  if oracle: #4b-sbm, return the first two blocks
    dense_num = nvecs[0] + nvecs[1]
    A_dense = A[:dense_num, :dense_num] #A[:n//2, :n//2]
    dense_IDs = np.array([True] * dense_num + [False] * (n - dense_num)) #mask out dense nodes - always a boolean array because we want to use ~dense_IDs to select sparse nodes!
    if power:
      assert feat is not None, "must pass in node feature if using PowerEmbed"
      powers = power_iterate(A_dense, feat[:dense_num, :], K=2)
      feat_dense = np.concatenate(powers, axis=1)
    else:
      feat_dense = emb_ASE(A_dense, k=k)

  else:
    assert kc is not None, "must specify the minimum node degree for kcore"
    if leader:
        localleader, leaders = IdentifyLocalLeaders(A, lamb=0.5)
        dense_IDs = np.array([False] * n)
        dense_IDs[leaders] = True
        A_dense = A[np.ix_(leaders, leaders)]
    else:
        nx_graph = nx.from_numpy_array(A)
        graph_dense = nx.k_core(nx_graph, k=kc)
        dense_nodes = list(graph_dense.nodes())
        dense_IDs = np.array([False] * n)
        dense_IDs[dense_nodes] = True
        A_dense = nx.to_numpy_array(graph_dense)
    if verbose:
      print(f"kcore graph size = {np.sum(dense_IDs)}")
    if power:
      assert feat is not None, "must pass in node feature if using PowerEmbed"
      powers = power_iterate(A_dense, feat[dense_IDs,:], K=2 )
      feat_dense = np.concatenate(powers, axis=1)
    else:
      feat_dense = emb_ASE(A_dense, k=k)

  #dense classify
  train_mask_dense = train_mask[dense_IDs]
  test_acc_dense, num_dense_test, Yhats_dense = feature_classify(feat_dense, Ys[dense_IDs], train_mask_dense)
  #sparse classify
  train_mask_sparse = train_mask[~dense_IDs]
  test_mask_sparse = ~train_mask[~dense_IDs]
  #message-passing
  num_dense_neighbors = A[np.ix_(~dense_IDs,dense_IDs)].sum(axis=1)
  num_dense_neighbors[num_dense_neighbors == 0] = -1
  assert  ks+degree+NI+LP <= 1
  if LP == True:
      adj = sp.csr_matrix(A)# + sp.eye(A.shape[0])
      # Row-column-normalize sparse matrix
      rowsum = np.array(adj.sum(1))
      r_inv = np.power(rowsum, -1 / 2).flatten()
      r_inv[np.isinf(r_inv)] = 0.
      r_mat_inv = sp.diags(r_inv)
      adj = r_mat_inv.dot(adj).dot(r_mat_inv)
      adj = sparse_mx_to_torch_sparse_tensor(adj)
      labels = np.array([0 if i < 0 else 1 for i in Yhats_dense])
      idx = dense_nodes
      K = 10
      alpha = 0.1
      y = label_propagation(adj, labels, idx, np.array(range(len(train_mask_dense))), K, alpha)
      Yhats = torch.argmax(y, dim=1)
      Yhats = np.array(Yhats, dtype=int)
      Yhats[Yhats == 0] = -1
      Yhats_sp = Yhats[~dense_IDs]
      if PT == True:
          y_soft_train = label_propagation(adj, labels,  # 只有dense nodes的labels
                                           np.array(dense_nodes)[train_mask_dense],  # 在所有nodes中的编号
                                           np.argwhere(train_mask_dense == True).reshape(-1),  # 在dense nodes中的编号
                                           K, alpha)
          y_soft_val = label_propagation(adj, labels,
                                         np.array(dense_nodes)[~train_mask_dense],
                                         np.argwhere(train_mask_dense == False).reshape(-1),
                                         K, alpha)
          model = PropTrain(nfeat=feat.shape[1],
                            nhid=64,
                            nclass=labels.max().item() + 1,
                            dropout=0,
                            epsilon=100,
                            mode=mode,
                            K=K,
                            alpha=alpha)
          optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
          stopping_args = Stop_args(patience=100, max_epochs=200)
          early_stopping = EarlyStopping(model, **stopping_args)
          for epoch in range(2000):
              loss_val, acc_val = train(epoch, model, optimizer, adj, feat, labels,
                                        np.array(dense_nodes)[train_mask_dense],
                                        np.argwhere(train_mask_dense==True).reshape(-1),
                                        np.array(dense_nodes)[~train_mask_dense],
                                        np.argwhere(train_mask_dense==False).reshape(-1),
                                        y_soft_train, y_soft_val, loss_decay=0.05, weight_decay=0.5, fast_mode=False)
              if early_stopping.check([acc_val, loss_val], epoch):
                  break
          print("Optimization Finished!")
          output = test(model, adj, feat, labels, idx, np.array(range(len(train_mask_dense))),
                        y, loss_decay=0.05)
          Yhats = torch.argmax(output, dim=1)
          Yhats = np.array(Yhats, dtype=int)
          Yhats[Yhats == 0] = -1
          Yhats_sp = Yhats[~dense_IDs]
  else:
      if ks==True:
        NI_index = NodeImportance(A, ks=True, prop_steps=prop_steps)
        message_fromDense = A[np.ix_(~dense_IDs, dense_IDs)] @ (Yhats_dense * NI_index[dense_IDs])
      elif degree==True:
        NI_index = NodeImportance(A, degree=True, prop_steps=prop_steps)
        message_fromDense = A[np.ix_(~dense_IDs, dense_IDs)] @ (Yhats_dense * NI_index[dense_IDs])
      elif NI == True:
        NI_index = NodeImportance(A, prop_steps=prop_steps)
        message_fromDense = A[np.ix_(~dense_IDs, dense_IDs)] @ (Yhats_dense * NI_index[dense_IDs])
      else:
        message_fromDense = A[np.ix_(~dense_IDs,dense_IDs)] @ Yhats_dense / num_dense_neighbors  # 1和-1哪个多
      #label prop:
      Yhats_sp = message_fromDense > 0
      Yhats_sp = Yhats_sp.astype(int)
      Yhats_sp[Yhats_sp == 0] = -1
  test_acc_sp = np.sum((Yhats_sp[test_mask_sparse] == Ys[~dense_IDs][test_mask_sparse]))
  num_sparse_test = len(Yhats_sp[test_mask_sparse])
  if verbose:
    print(f"dense acc = {test_acc_dense:.2f}, num_dense_test = {num_dense_test}, sparse acc = {(test_acc_sp/num_sparse_test):.2f}, num_sparse_test = {num_sparse_test}")
  test_acc = (test_acc_dense * num_dense_test + test_acc_sp)/(len(Ys[~train_mask]))
  #label as features:
  # test_acc_sp_feat, num_sparse_test, _ = feature_classify(message_fromDense.reshape(-1,1), Ys[~dense_IDs], train_mask_sparse )
  # test_acc_feat = (test_acc_dense * num_dense_test + test_acc_sp_feat * num_sparse_test)/(len(Ys[~train_mask]))
  return test_acc, 0


def JaccardDistance(A_i, A_j):
    assert len(A_i) == len(A_j), "A_i and A_j should have same lengths"
    i_neigh = set(np.argwhere(A_i == 1).reshape(-1))
    j_neigh = set(np.argwhere(A_j == 1).reshape(-1))
    inter = i_neigh.intersection(j_neigh)
    union = i_neigh.union(j_neigh)
    return (len(union)-len(inter))/len(union)  # len(union)-


def NodeImportance(A, prop_steps=3, ks=False, degree=False):
    nx_graph = nx.from_numpy_array(A)
    #  ks index
    core_num = nx.core_number(nx_graph)
    if degree == True:
        NI = A.sum(axis=1)
    elif ks == True:
        NI = np.array(list(core_num.values()))
    else:
        # Signal propagation amount
        S = (A + np.eye(A.shape[0]))
        for i in range(prop_steps - 1):
            S = (A+np.eye(A.shape[0])) @ S
        NI = []
        for i in range(A.shape[0]):
            J_distance = []
            for j in range(A.shape[0]):
                J_distance.append(JaccardDistance(A[i], A[j]))
            NI_i = S[i, i] * core_num[i] * sum(A[i] * np.array(J_distance) * np.array(list(core_num.values())))
            NI.append(NI_i)
    return np.array(NI)


def JaccardSimilarityAndDistance(A_i, A_j, i, j):
    assert len(A_i) == len(A_j), "A_i and A_j should have same lengths"
    i_neigh = set(np.argwhere(A_i == 1).reshape(-1))
    i_neigh.add(i)
    j_neigh = set(np.argwhere(A_j == 1).reshape(-1))
    j_neigh.add(j)
    inter = i_neigh.intersection(j_neigh)
    union = i_neigh.union(j_neigh)
    return [len(inter) / len(union), len(union) / len(inter)]


def Leadership(A):
    LS_all = []
    for i in range(A.shape[0]):
        neigh = np.argwhere(A[i] == 1).reshape(-1)
        LS = 0
        for j in neigh:
            LS += JaccardSimilarityAndDistance(A[i], A[j], i, j)[0]
        LS_all.append(LS)
    return np.array(LS_all)


def IdentifyLocalLeaders(A, lamb=0.5):
    localleader = []
    leaders = []
    LS = Leadership(A)
    for i in range(A.shape[0]):
        i_neigh = set(np.argwhere(A[i] == 1).reshape(-1))
        EC_i_neigh = []
        AF_i_neigh = []
        leader_i = i
        force = 0
        for j in i_neigh:
            # Edge compactness
            SimAndDis = JaccardSimilarityAndDistance(A[i], A[j], i, j)
            EC_i_j = SimAndDis[0]
            j_neigh = set(np.argwhere(A[j] == 1).reshape(-1))
            CN = i_neigh.intersection(j_neigh)
            UN = i_neigh - CN
            for t in CN:
                EC_i_j += JaccardSimilarityAndDistance(A[i], A[t], i, t)[0] * JaccardSimilarityAndDistance(A[t], A[j], t, j)[0]
            for t in UN:
                tmp = JaccardSimilarityAndDistance(A[t], A[j], t, j)[0]
                if tmp >= lamb:
                    EC_i_j += JaccardSimilarityAndDistance(A[i], A[t], i, t)[0] * tmp
                else:
                    EC_i_j += JaccardSimilarityAndDistance(A[i], A[t], i, t)[0] * (tmp - lamb)
            EC_i_neigh.append(EC_i_j)
            # Attractive force
            AF_j_i = sum(A[j]) / sum(A[i]) * LS[j] / (SimAndDis[1]**2)
            AF_i_neigh.append(AF_j_i)

        for index, j in enumerate(i_neigh):
            if (LS[j] > LS[i]) & (EC_i_neigh[index] >= 0) & (AF_i_neigh[index] > force):
                force = AF_i_neigh[index]
                leader_i = j
        localleader.append(leader_i)
        if leader_i == i:
            leaders.append(i)
    for i in range(A.shape[0]):
        leader = localleader[i]
        while leader != localleader[leader]:
            leader = localleader[leader]
        localleader[i] = leader
    return np.array(localleader), np.array(leaders)


def leader_classify(A, nvecs, Ys, train_mask, oracle=True, k=2, verbose=False, power=False, feat=None, lamb=0.5):
  '''
  Graph = leader(dense) + follower(sparse)
  '''
  n = A.shape[0]
  if oracle: #4b-sbm, return the first two blocks
    dense_num = nvecs[0] + nvecs[1]
    A_dense = A[:dense_num, :dense_num] #A[:n//2, :n//2]
    dense_IDs = np.array([True] * dense_num + [False] * (n - dense_num)) #mask out dense nodes - always a boolean array because we want to use ~dense_IDs to select sparse nodes!
    if power:
      assert feat is not None, "must pass in node feature if using PowerEmbed"
      powers = power_iterate(A_dense, feat[:dense_num, :], K=2)
      feat_dense = np.concatenate(powers, axis=1)
    else:
      feat_dense = emb_ASE(A_dense, k=k)

  else:
    localleader, leaders = IdentifyLocalLeaders(A, lamb=lamb)
    dense_IDs = np.array([False] * n)
    dense_IDs[leaders] = True
    if verbose:
      print(f"kcore graph size = {np.sum(dense_IDs)}")
    A_dense = A[np.ix_(leaders, leaders)]
    if power:
      assert feat is not None, "must pass in node feature if using PowerEmbed"
      powers = power_iterate(A_dense, feat[dense_IDs,:], K=2 )
      feat_dense = np.concatenate(powers, axis=1)
    else:
      feat_dense = emb_ASE(A_dense, k=k)

  # dense classify
  train_mask_dense = train_mask[dense_IDs]
  test_acc_dense, num_dense_test, Yhats_dense = feature_classify(feat_dense, Ys[dense_IDs], train_mask_dense)
  # sparse classify
  train_mask_sparse = train_mask[~dense_IDs]
  test_mask_sparse = ~train_mask[~dense_IDs]

  # label propagation
  Yhats = np.array([0] * A.shape[0])
  Yhats[dense_IDs] = Yhats_dense
  for i in range(len(Yhats)):
      Yhats[i] = Yhats[localleader[i]]
  Yhats_sp = Yhats[~dense_IDs]

  test_acc_sp = np.sum((Yhats_sp[test_mask_sparse] == Ys[~dense_IDs][test_mask_sparse]))
  num_sparse_test = len(Yhats_sp[test_mask_sparse])
  if verbose:
    print(f"dense acc = {test_acc_dense:.2f}, num_dense_test = {num_dense_test}, sparse acc = {(test_acc_sp/num_sparse_test):.2f}, num_sparse_test = {num_sparse_test}")
  test_acc = (test_acc_dense * num_dense_test + test_acc_sp)/(len(Ys[~train_mask]))
  # #label as features:
  # test_acc_sp_feat, num_sparse_test, _ = feature_classify(message_fromDense.reshape(-1,1), Ys[~dense_IDs], train_mask_sparse )
  # test_acc_feat = (test_acc_dense * num_dense_test + test_acc_sp_feat * num_sparse_test)/(len(Ys[~train_mask]))
  return test_acc, 0


def MC_run(n, nvecs, p_d, q_d, p_s, q_s, a, b, k=2, num_runs=30, oracle=True, kc=20, verbose=False):
    '''
    nvecs = [size_1, size_2, size_3, size_4], must sum to n
    '''
    # train mask
    torch.manual_seed(0)
    train_pct = 0.5
    Ys = np.array([1] * nvecs[0] + [-1] * nvecs[1] + [1] * nvecs[2] + [-1] * nvecs[3])
    # MC run
    # {'ASE': [], 'Cov(X)': [], 'A_X': [], 'MP-1': [], 'MP-2': [],
    results = {'ASE': [], 'Cov(X)': [], 'A_X': [], 'Power_2': [], 'MP-1': [], 'MP-2': [],
               'decomp(ASE)': [], 'decomp(Power)': [], 'KS(Power)': [], 'Degree(Power)': [], 'NI(Power)': [],
               'Leader(Power)': [],
               'LP(Power)': [], 'PT(Power)': []} #, 'NI(Power)_3': [], 'NI(Power)_4': []}
    #, 'NI(Power)': [], 'leader(Power)': []}

    for run in range(num_runs):
        print(run)
        A = construct_graph(n, nvecs, p_d, q_d, p_s, q_s, a, b, seed=run)
        # add self-loop
        # A = A + np.eye(n)
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
        SGC_1 = emb_SGC(A, features, k=k, n_layer=1)
        SGC_2 = emb_SGC(A, features, k=k, n_layer=2)

        for name, emb in zip(['ASE', 'Cov(X)', 'A_X', 'MP-1', 'MP-2'],
                             [ASE_feat, cov_feat, A_X_feat, SGC_1, SGC_2]):
            acc, _, _ = feature_classify(emb, Ys, train_mask_all)
            results[name].append(acc)
            # graph decomp

        powers = power_iterate(A, features, K=2)
        power_emb = np.concatenate(powers, axis=1)
        acc_power, _, _ = feature_classify(power_emb, Ys, train_mask_all)
        results['Power_2'].append(acc_power)

        decomp_acc, _ = kcore_classify(A, nvecs, Ys, train_mask_all, oracle=oracle, kc=kc, k=k, verbose=verbose)
        results['decomp(ASE)'].append(decomp_acc)
        decomp_acc_pow, _ = kcore_classify(A, nvecs, Ys, train_mask_all, oracle=oracle, kc=kc, k=k, verbose=verbose,
                                           power=True, feat=features)
        results['decomp(Power)'].append(decomp_acc_pow)

        decomp_acc_pow_ks, _ = kcore_classify(A, nvecs, Ys, train_mask_all, oracle=oracle, kc=kc, k=k, verbose=verbose,
                                           power=True, feat=features, ks=True)
        results['KS(Power)'].append(decomp_acc_pow_ks)
        decomp_acc_pow_d, _ = kcore_classify(A, nvecs, Ys, train_mask_all, oracle=oracle, kc=kc, k=k, verbose=verbose,
                                              power=True, feat=features, degree=True)
        results['Degree(Power)'].append(decomp_acc_pow_d)
        decomp_acc_pow_NI_2, _ = kcore_classify(A, nvecs, Ys, train_mask_all, oracle=oracle, kc=kc, k=k, verbose=verbose,
                                             power=True, feat=features, NI=True, prop_steps=2)
        results['NI(Power)'].append(decomp_acc_pow_NI_2)

        decomp_acc_pow_leader, _ = kcore_classify(A, nvecs, Ys, train_mask_all, oracle=oracle, kc=kc, k=k,
                                                  verbose=verbose,
                                                  power=True, feat=features, leader=True)
        results['Leader(Power)'].append(decomp_acc_pow_leader)

        decomp_acc_pow_LP, _ = kcore_classify(A, nvecs, Ys, train_mask_all, oracle=oracle, kc=kc, k=k,
                                                verbose=verbose,
                                                power=True, feat=features, LP=True, prop_steps=2)
        results['LP(Power)'].append(decomp_acc_pow_LP)
        decomp_acc_pow_PT, _ = kcore_classify(A, nvecs, Ys, train_mask_all, oracle=oracle, kc=kc, k=k,
                                              verbose=verbose,
                                              power=True, feat=features, LP=True, prop_steps=2, PT=True)
        results['PT(Power)'].append(decomp_acc_pow_PT)


        # decomp_acc_pow_NI_3, _ = kcore_classify(A, nvecs, Ys, train_mask_all, oracle=oracle, kc=kc, k=k, verbose=verbose,
        #                                       power=True, feat=features, NI=True, prop_steps=3)
        # results['NI(Power)_3'].append(decomp_acc_pow_NI_3)
        # decomp_acc_pow_NI_4, _ = kcore_classify(A, nvecs, Ys, train_mask_all, oracle=oracle, kc=kc, k=k, verbose=verbose,
        #                                       power=True, feat=features, NI=True, prop_steps=4)
        # results['NI(Power)_4'].append(decomp_acc_pow_NI_4)
        # leader_acc_pow, _ = leader_classify(A, nvecs, Ys, train_mask_all, oracle=oracle, k=k, verbose=verbose,
        #                                     power=True, feat=features, lamb=0.3)
        # results['leader(Power)'].append(leader_acc_pow)

    return results


def plot_result(results_all, n, nvecs, p_d, q_d, p_s, q_s):
  fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 6), sharey=True, sharex=True, dpi=150)

  for i, (key, results) in enumerate(results_all.items()):
    row, col = int(i / 3), i % 3
    df = pd.DataFrame.from_dict(results)
    df.boxplot(grid=False, rot=45, fontsize=12, ax=axs[row, col])  # [row, col]
    axs[row, col].set_ylabel('Accuracy')
    a, b = key
    axs[row, col].set_title(f"a={a:.3f}, b={b:.3f}")

  #fig.suptitle(f"4-block 2-class symmetric SBM, n={n}, p={p_d:.2f}, q={q_d:.2f}, p'={p_s:.3f}, q'={q_s:.3f}, MC_runs=20", fontsize=12, y=1.02)
  #fig.suptitle(f"sizes=({nvecs})", y=1.02)
  fig.tight_layout()
  plt.show()


if __name__ == "__main__":
    n = 300
    nvecs = [n // 4, n // 4, n // 4, n // 4]
    # opt_a, opt_b, opt_pp, opt_qq = 2/n, 1/(2*n)
    # trick: a, b must also be sparse in order for kcore to outperform ASE
    configs = [(1 / 25, 1 / 50), (1 / 15, 1 / 30),
               (2 / 25, 1 / 25), (2 / 15, 1 / 15),
               (1 / 5, 1 / 10), (1 / 4, 1 / 8)]  # (1/10, 1/20), (1 / 15, 1 / 30), (1 / 25, 1 / 50) 30 50
    p_d, q_d = 1 / 3, 1 / 4  # within class
    p_s, q_s = 1 / 30, 1 / 40  # between class
    results_all_34 = {}
    # oracle setting: where we KNOW where the dense subgraph is!
    # for (a, b) in configs:
    #     results_all_34[(a, b)] = MC_run(n, nvecs, p_d, q_d, p_s, q_s, a, b, num_runs=30, oracle=True, verbose=False)
    #
    # plot_result(results_all_34, n, nvecs, p_d, q_d, p_s, q_s)

    # Practical setting: we do not know the dense subgraph, and need to estimate it from the actual graph using kcore subgraph
    # estimated the threshold for kcore minimum degree: pretty robust (from kc=15 ~ 20 all work fine)
    results_all_34_est_15 = {}
    for (a, b) in configs:
        results_all_34_est_15[(a, b)] = MC_run(n, nvecs, p_d, q_d, p_s, q_s, a, b, num_runs=30, oracle=False, k=2,
                                               kc=15, verbose=False)
    import pickle
    with open("results_all_34_est_15.pkl", "wb") as tf:
        pickle.dump(results_all_34_est_15, tf)
    plot_result(results_all_34_est_15, n, nvecs, p_d, q_d, p_s, q_s)