import math
import random
import os
import numpy as np
import numpy.random as npr
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import WebKB, WikipediaNetwork, Planetoid, Amazon, Coauthor, Actor
from torch_geometric.utils import to_networkx, homophily
import time
from scipy.sparse.linalg import eigsh, eigs
import pandas as pd
import seaborn as sns
import copy
#import common functions
from power import power_iterate, run_exp
import argparse

#reproducibility
np.random.seed(0)
torch.manual_seed(0)


def extract_data_subgraph(data, kcoreIDs, k):
  '''
  data: pytorch geometric data (original)
  kcoreIDs: nodes to be included in the induced subgraph
  k: number of principal components used for the features
  '''
  X_kc = data.x[kcoreIDs].numpy()
  evaluesX, evectorsX = eigsh(X_kc @ X_kc.T, k=k)
  feat_X = torch.FloatTensor(evectorsX[:, :k])
  data_kc = copy.deepcopy(data)
  data_kc.x = data.x[kcoreIDs]
  data_kc.y = data.y[kcoreIDs]
  data_kc.train_mask = data.train_mask[kcoreIDs].bool()
  data_kc.val_mask = data.val_mask[kcoreIDs].bool()
  data_kc.test_mask = data.test_mask[kcoreIDs].bool()
  return data_kc, feat_X

def get_kcore_yhats(graph_kCORE_tensor, data, feat_X, num_hops=10, verbose=False, rw=False, device='cuda:0'): #, labels, train_mask, val_mask, test_mask, num_hops=7):
  '''
  Return test_acc on kcore graph, and predicted dense node labels (with training label unchanged)
  '''
  #PowerNet
  Powers = power_iterate(graph_kCORE_tensor, torch.FloatTensor(feat_X), K=num_hops, upper_tri=False, rw=rw)
  test_acc, yhats, val_acc = run_exp(Powers, data, K=num_hops+1,all_iter='all', individual=True, device=device)
  runs = data.train_mask.shape[1]
  #correct predictions on the train labels
  for k in range(1,num_hops+1):
    for run in range(runs):
      train_mask_run = data.train_mask[:, run]
      #print(yhats[k][run].shape, train_mask_run.shape)
      yhats[k][run][train_mask_run] = data.y[train_mask_run]
      if verbose:
        print(f"test acc of KCORE graph = {test_acc[k][run]:.2f}")
  return test_acc, yhats, val_acc

def get_sparse_subgraph(n, nx_graph, kcoreIDs):
  '''
  Given the original graph and the dense subgraph, extract nodes in the original graph that have at least one dense neighbor
  Return: list of sparse nodes, dict of their dense neighbors
  '''
  sparse_IDs = list(set(np.arange(n)) - set(kcoreIDs))
  sparse_subgraph = []
  dense_neighbors = {}
  for node_sp in sparse_IDs:
    if any(neighbor in kcoreIDs for neighbor in nx_graph.neighbors(node_sp)):
      sparse_subgraph.append(node_sp)
      dense_neighbors[node_sp] = list(set(nx_graph.neighbors(node_sp)).intersection(kcoreIDs))
  return sparse_subgraph, dense_neighbors

def predict_sparse_nodes(data, n, test_acc_kc, val_acc_kc, yhats_kc, kcoreIDs, sparse_subgraph, dense_neighbors):
  '''
  Perform majority votes among their dense neighbors on sparse nodes
  '''
  #mapping
  ID_map = {id_whole:id_dense for id_dense,id_whole in enumerate(kcoreIDs)}
  test_acc_sp = []
  test_acc_sp_num = []
  test_acc_den = []
  test_acc_den_num = []
  nids = torch.arange(n)
  runs = data.train_mask.shape[1]

  for run in range(runs):
    maxValue, keyMaxValue = max((val_acc_kc[key][run],key) for key in val_acc_kc)
    test_acc_den.append(test_acc_kc[keyMaxValue][run])
    Yhats_dense = yhats_kc[keyMaxValue][run].numpy()
    test_nids = nids[data.test_mask[:,run]]
    test_sp_ids = list(set(sparse_subgraph).intersection(test_nids.tolist()))
    test_acc_den_num.append(len(set(kcoreIDs).intersection(test_nids.tolist())))

    #majority vote among dense neighbors
    Yhats_test_sp = []
    for node_sp in test_sp_ids: #sparse_subgraph:
      node_neigh_id_sp = [ID_map[i] for i in dense_neighbors[node_sp]]
      node_neigh_Yhats = Yhats_dense[node_neigh_id_sp]
      values, counts = np.unique(node_neigh_Yhats, return_counts=True)
      ind = np.argmax(counts)
      Yhats_test_sp.append(values[ind])
    node_num = len(test_sp_ids)
    test_acc_sp_num.append(node_num)
    acc = (torch.LongTensor(Yhats_test_sp) == data.y[test_sp_ids]).sum()/node_num
    test_acc_sp.append(acc.item())
  
  return test_acc_den, test_acc_den_num, test_acc_sp, test_acc_sp_num

def run_graph_decomp(data_ori, self_loops=True, k=10, kc=3, num_hops=10, rw=True, device='cuda:0'):
  '''
  1) Decompose graph into dense (kcore) + sparse, where each node in the sparse subgraph has at least one dense node neighbor
  2) Run PowerEmbed w/ classifier on dense to obtain yhats for dense nodes
  3) Use majority vote among dense neighbors to obtain yhats for sparse nodes (assume homophily)
  Accuracy: 1/(num_dense + num_sparse) * (test_acc[dense]*num_dense + test_acc[sparse]*num_sparse)
  [TODO: ITERATION ON 3 to label all sparse nodes in the graph?!]
  '''
  test_dense, test_all = [], []
  #preprocess data
  data = copy.deepcopy(data_ori)
  nx_graph_ori = to_networkx(data,to_undirected=True, remove_self_loops=True)
  n = nx_graph_ori.number_of_nodes()

  #decomp
  nx_graph = nx.k_core(nx_graph_ori, k=kc) #nx.k_core(nx_graph, k=kc)
  kcoreIDs = list(nx_graph.nodes())
  data_kc, feat_X = extract_data_subgraph(data, kcoreIDs, k)
  A = nx.to_numpy_array(nx_graph)
  A_tensor = torch.FloatTensor(A) 
  if self_loops:
    A_tensor = A_tensor + torch.eye(nx_graph.number_of_nodes())
  # Xouter = data.x.numpy() @ data.x.numpy().T
  print(f'A: {A.shape}, feat_X: {feat_X.shape}')

  #dense node prediction
  test_acc_kc, yhats_kc, val_acc_kc = get_kcore_yhats(A_tensor, data_kc, feat_X, num_hops, rw=rw, device=device)
  #test_acc_den = np.array(test_acc_kc[hop])#np.array([np.array(acc) for key, acc in test_acc_kc.items()])

  #sparse node prediction
  sparse_subgraph, dense_neighbors = get_sparse_subgraph(n, nx_graph_ori, kcoreIDs)
  test_acc_den, test_acc_den_num, test_acc_sp, test_acc_sp_num = predict_sparse_nodes(data_ori, n, test_acc_kc, val_acc_kc, yhats_kc, \
                                                                                      kcoreIDs, sparse_subgraph, dense_neighbors)

  #overall acc
  test_overall = [ (acc_dense*num_dense + acc_sp * num_sp)/(num_dense + num_sp) \
                for acc_dense, num_dense, acc_sp, num_sp in zip(test_acc_den, test_acc_den_num, test_acc_sp, test_acc_sp_num)]

  return test_overall, test_acc_den, test_acc_sp, kcoreIDs, sparse_subgraph


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    data_all = Planetoid(root=args.data_path, name=args.datasets, split='geom-gcn') 
    data = data_all[0] 
    nx_graph = to_networkx(data, to_undirected=True, remove_self_loops=True)
    dim_kc = 10
    dim_full = 100
    runs = data.train_mask.shape[1]
    for kc in [3,4]:
      test_overall_kc, test_den_kc, test_sp_kc, kcoreIDs, sparseIDs = run_graph_decomp(data, self_loops=True, k=dim_kc, kc=kc, num_hops=10, rw=True, device=device)
      print(f"kc={kc}: test_mean = {(np.array(test_overall_kc).mean()):.4f}, test_den = {(np.array(test_den_kc).mean()):.4f}, \
           test_sp = {(np.array(test_sp_kc).mean()):.4f} ")
      print(f"kc={kc}: test_std = {(np.array(test_overall_kc).std()/np.sqrt(10)):.4f}, test_den_std = {(np.array(test_den_kc).std()/np.sqrt(10)):.4f}, \
           test_sp_std = {(np.array(test_sp_kc).std()/np.sqrt(10)):.4f} ")
      #benchmark: powerembed on the entire subgraph
      nids_sub = kcoreIDs + sparseIDs
      data_sub, feat_X_sub = extract_data_subgraph(data, nids_sub, k=dim_full)
      nx_graph_sub = nx_graph.subgraph(nids_sub)
      A_sub = nx.to_numpy_array(nx_graph_sub)
      #dense node prediction
      test_acc_all, _, val_acc_all = get_kcore_yhats(torch.FloatTensor(A_sub)+torch.eye(data_sub.num_nodes), data_sub, feat_X_sub, num_hops=10, rw=True)
      test_acc_bk = []
      for run in range(runs):
        maxValue, keyMaxValue = max((val_acc_all[key][run],key) for key in val_acc_all)
        test_acc_bk.append(test_acc_all[keyMaxValue][run])
      print(f"whole: test_mean = {(np.array(test_acc_bk).mean()):.4f}, test_std = {(np.array(test_acc_bk).std()/np.sqrt(10)):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph_decomposition_planetoid")
    parser.add_argument("--datasets", type=str, default="wiki", help="datasets: Cora or CiteSeer")
    parser.add_argument("--data_path", type=str, default="./dataset/", help="dataset folder path")
    parser.add_argument("--result_path", type=str, default="./result_noUT/", help="dataset folder path")
    parser.add_argument("--R", type=int, default=10,help="number of hops")
    #parser.add_argument("--k", type=int, default=100,help="top-k eigenvectors of cov(X)")
    parser.add_argument("--loop", action='store_true', help="add self loop to the graph")
    parser.add_argument("--pow_only", action='store_true', help="only run experiments for PowerEmbed")
    parser.add_argument("--kcore", action='store_true', help="run on kcore subgraph")
    parser.add_argument("--kcore_size", type=int, default=4, help="kcore subgraph degree")

    args = parser.parse_args()