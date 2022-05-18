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
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import WebKB, WikipediaNetwork, Planetoid, Amazon, Coauthor, Actor, SNAPDataset
from torch_geometric.utils import to_networkx, homophily
import time
from scipy.sparse.linalg import eigsh, eigs
import pandas as pd
import seaborn as sns
import copy
import argparse

#reproducibility
np.random.seed(0)
torch.manual_seed(0)

###inception 
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
    def __init__(self, in_feats, hidden, out_feats, num_hops, n_layers, dropout, input_drop, subnet=True):
        '''
        in_feats, hidden: lists of input features/hidden dimension
        '''
        super(SIGN_POWER, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.input_drop = nn.Dropout(input_drop)
        self.subnet = subnet
        if self.subnet: #MLP subnets for each power features
          self.inception_ffs = nn.ModuleList()
          for hop in range(num_hops):
            self.inception_ffs.append(FeedForwardNet(in_feats[hop], hidden[hop], hidden[hop], n_layers, dropout))
          hidden_sum = np.array(hidden).sum()
          self.project = FeedForwardNet(hidden_sum, hidden_sum, out_feats,n_layers, dropout)
        else: #MLP for last iterate features
          self.project = FeedForwardNet(in_feats, hidden, out_feats, n_layers, dropout)

    def forward(self, feats):
        if self.subnet:
          feats = [self.input_drop(feat) for feat in feats]
          hidden = []
          #concatenate outputs from each subnets (size n by d*num_hops), followed by a MLP
          for i, (feat, ff) in enumerate(zip(feats, self.inception_ffs)):
              hidden.append(ff(feat))
          out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        else: #one linear output layer
          #feats = torch.cat(feats, dim=-1) 
          out = self.project(feats)
        return out

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()

def train(model, feats, labels, loss_fcn, optimizer, train_loader, feat_list=True):
    model.train()
    device = labels.device
    for batch in train_loader:
        if feat_list:
            batch_feats = [x[batch].to(device) for x in feats]
        else:
            batch_feats = feats[batch].to(device)
        loss = loss_fcn(model(batch_feats), labels[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, feats, labels, test_loader, train_nid, val_nid, test_nid, feat_list=True, individual=False):
    model.eval()
    device = labels.device
    preds = []
    for batch in test_loader:
        if feat_list:
            batch_feats = [feat[batch].to(device) for feat in feats]
        else:
            batch_feats = feats[batch].to(device)
        preds.append(torch.argmax(model(batch_feats), dim=-1))
    # Concat mini-batch prediction results along node dimension
    preds = torch.cat(preds, dim=0)
    train_res = (preds[train_nid] == labels[train_nid]).sum()/len(train_nid)
    val_res = (preds[val_nid] == labels[val_nid]).sum()/len(val_nid)
    test_res = (preds[test_nid] == labels[test_nid]).sum()/len(test_nid)
    if individual: #can potentially add preds[train_nid] == labels[train_nid], preds[val_nid] == labels[val_nid], 
      return train_res, val_res, test_res, preds
    else:
      return train_res, val_res, test_res

def power_iterate(g, feat, K, truncate=None, normalize=True, upper_tri=False, rw=False):
    """
    Precompute power-iterated features: sparse g
    g: torch tensor (n by n)
    Return: a list with length-K of feature matrices, each with shape (n by d)
    If truncate: truncate the list to keep only the first few and last few iterates
    If include_feat: include the original features
    If upper_tri: use the upper triangular mask in the normalization
    If rw: use D^{-1}A instead of A as the graph operators
    """
    powers = [feat]
    if rw:
      Dinv = 1 / g.sum(axis=1) 
      g = g * Dinv[:, None] 
    g = g.to_sparse()
    for iter in range(K):
      #message passing
      tildeU = g @ feat 
      #normalization
      if normalize:
        if upper_tri:
          normalizer = torch.linalg.pinv(torch.triu(tildeU.T @ tildeU))
        else:
          normalizer = torch.linalg.pinv(tildeU.T @ tildeU)
        feat = tildeU @ normalizer #scaled eigenvector (arbitrary rotated if upper_tri is False)
        col_norm = torch.linalg.norm(feat, dim=0)
        powers.append(feat / col_norm)
        #feat = feat / col_norm #[BUG: if we normalize at every t, we may not get the same convergence to scaled eigenvectors?! CHECK]
      else:
        feat = tildeU
        powers.append(feat)
    return powers

def run_model(model, feats, labels, train_loader, test_loader, train_nid, val_nid, test_nid,  
                  lr, weight_decay, num_epochs, eval_every, feat_list, verbose=False, individual=False):
    #optim
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    # Start training
    best_epoch = 0
    best_val = 0
    best_test = 0
    if individual:
        best_indiv = 0

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train(model, feats, labels, loss_fcn, optimizer, train_loader, feat_list)

        if epoch % eval_every == 0:
            with torch.no_grad():
                acc = test(model, feats, labels, test_loader, train_nid, val_nid, test_nid, feat_list, individual=individual)
            end = time.time()
            log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
            log += "Acc: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(*acc)
            if verbose:
              print(log)
            if acc[1] > best_val:
                best_epoch = epoch
                best_val = acc[1]
                best_test = acc[2]
                if individual: #yhats for all dense nodes
                    best_indiv = acc[3]
    if verbose:
      print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
        best_epoch, best_val, best_test))
    
    if individual:
      return best_val, best_test, best_indiv
    else:
      return best_val, best_test

def run_exp(list_of_feats, data, K, device='cuda:0', eval_every = 1,
            lr=1e-2, n_layer=2, dropout=0.5, all_iter='all', verbose=False, individual=False):
  '''
  list_of_feats: a list of feature where each one obtained from power iteration (w/wo normalization)
  data: pytorch geometric data class
  K: max iteration
  all_iter: 'all' - use all iterations; 'last' - use the last iteration; 'first_last" - use the first (i.e. input feat) and the last iter
  '''
  acc_POWER = {}
  if individual:
    pred_POWER, val_POWER = {}, {}
  #hyperparams
  if all_iter == 'all': #list of feats
    in_size = [list_of_feats[k].shape[1] for k in range(len(list_of_feats))]
    num_hidden = [input_size * 2 for input_size in in_size]
    feat_list = True
  elif all_iter == 'last': 
    in_size = list_of_feats[0].shape[1]
    num_hidden = in_size * 2
    feat_list = False
  else:
    in_size = [list_of_feats[0].shape[1]] * 2
    num_hidden = [list_of_feats[0].shape[1]*2] * 2
    feat_list = True
  num_classes = len(torch.unique(data.y))
  labels = data.y.to(device)
  num_epochs = 100
  weight_decay = 0
  runs = data.train_mask.shape[1]
  #print('Total runs=', runs)
  #results
  for k in range(1,K):
    if all_iter == 'all':
      pow_k = list_of_feats[:(k+1)] #ultimate BUG!! all_iter starts with using feature only
    elif all_iter == 'last':
      pow_k = list_of_feats[k] 
    else: #first and last
      pow_k = [list_of_feats[0]] + [list_of_feats[k]]
      #print(k, len(pow_k))

    acc_POWER[k] = []
    if individual:
      pred_POWER[k] = []
      val_POWER[k] = []
    
    for run in range(runs):
      train_mask = data.train_mask[:,run]
      val_mask = data.val_mask[:,run]
      test_mask = data.test_mask[:, run]
      nids = torch.arange(data.num_nodes)
      train_nid = nids[train_mask]
      val_nid = nids[val_mask]
      test_nid = nids[test_mask]
      train_loader = torch.utils.data.DataLoader(train_nid, batch_size=10000, shuffle=True, drop_last=False)
      test_loader = torch.utils.data.DataLoader(torch.arange(data.num_nodes), batch_size=100000,shuffle=False, drop_last=False)
      if all_iter == 'all':
      #print(f"all_iter length = {len(pow_k)}, in_size length = {len(in_size[:(k+1)])}")
        model = SIGN_POWER(in_size[:(k+1)], num_hidden[:(k+1)], num_classes, num_hops=k+1, n_layers = n_layer, dropout=dropout, input_drop=0, subnet=True)
      elif all_iter == 'last':
        model = SIGN_POWER(in_size, num_hidden, num_classes, num_hops=1, n_layers = n_layer, dropout=dropout, input_drop=0, subnet=False)
      else:
        model = SIGN_POWER(in_size, num_hidden, num_classes, num_hops=2, n_layers = n_layer, dropout=dropout, input_drop=0, subnet=True)
      model = model.to(device)
      if individual:
        best_val, best_test, best_indiv = run_model(model, pow_k, labels, train_loader, test_loader, train_nid, val_nid, test_nid,  
            lr, weight_decay, num_epochs, eval_every, feat_list=feat_list, verbose=verbose, individual=True)   
        pred_POWER[k].append(best_indiv.cpu()) 
        val_POWER[k].append(best_val.cpu().numpy())           
      else:
        best_val, best_test = run_model(model, pow_k, labels, train_loader, test_loader, train_nid, val_nid, test_nid,  
            lr, weight_decay, num_epochs, eval_every, feat_list=feat_list, verbose=verbose)
      acc_POWER[k].append(best_test.cpu().numpy())
  if individual:
    return acc_POWER, pred_POWER, val_POWER
  else:
    return acc_POWER

def run_spectral(data, A, Xouter, k=10, lr=1e-2, n_layer=2, dropout=0.5, device='cuda:0', verbose=False):
  '''
  data: pytorch geometric graph data class
  A: graph adajacency
  Xouter: cov(X)
  k: embedding dimension
  undirected: indicate A is symmetric or not
  NOTE: for benchmark, use unscaled ASE (i.e. pure eigenvectors)
  '''
  acc_X, acc_ASE, acc_X_ASE = [], [], []
  #u, s, vh = np.linalg.svd(A)
  evalues, evectors = eigsh(A, k=k)
  feat_ASE = torch.FloatTensor(evectors[:, :k]) #* s[:k]
  print('ASE:', feat_ASE.shape)

  #ux, sx, vhx = np.linalg.svd(Xouter, hermitian=True)
  evaluesX, evectorsX = eigsh(Xouter, k=k)
  feat_X = torch.FloatTensor(evectorsX[:, :k])
  print('node cov:', feat_X.shape)

  num_classes = len(torch.unique(data.y))
  labels = data.y.to(device)
  num_epochs = 100
  eval_every = 1
  weight_decay = 0
  runs = data.train_mask.shape[1]
  #print('Total runs=', runs)
  for run in range(runs):
    #print(f"run={run}")
    train_mask = data.train_mask[:,run]    
    val_mask = data.val_mask[:,run]
    test_mask = data.test_mask[:, run]
    nids = torch.arange(data.num_nodes)
    train_nid = nids[train_mask]
    val_nid = nids[val_mask]
    test_nid = nids[test_mask]
    train_loader = torch.utils.data.DataLoader(train_nid, batch_size=10000, shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(torch.arange(data.num_nodes), batch_size=100000,shuffle=False, drop_last=False)

    for feat, result in zip([feat_X, feat_ASE], [acc_X, acc_ASE]):
      in_size = feat.shape[1]
      num_hidden = in_size * 2
      model = FeedForwardNet(in_size, num_hidden, num_classes, n_layers=n_layer, dropout=dropout) 
      model = model.to(device)
      best_val, best_test = run_model(model, feat, labels, train_loader, test_loader, train_nid, val_nid, test_nid,  
                                          lr, weight_decay, num_epochs, eval_every, feat_list=False, verbose=verbose)
      result.append(best_test.cpu().numpy())
    #both X and ASE
    in_size = [k] * 2
    num_hidden = [2*k] *2
    model = SIGN_POWER(in_size, num_hidden, num_classes, num_hops=2, n_layers = n_layer, dropout=dropout, input_drop=0, subnet=True)
    model = model.to(device)
    best_val, best_test = run_model(model, [feat_X, feat_ASE], labels, train_loader, test_loader, train_nid, val_nid, test_nid,  
                                        lr, weight_decay, num_epochs, eval_every, feat_list=True,verbose=verbose)
    acc_X_ASE.append(best_test.cpu().numpy())      

  return np.array(acc_X), np.array(acc_ASE), np.array(acc_X_ASE), feat_X


def run_data(data, self_loops=False, undirected=True, k=10, num_hops=10, eval_every=1,
             upper_tri=False, rw=True, kc=None, individual=False, power_only=False, device='cuda:0'):
  '''
  data: pytorch geometric data
  undirected: If true, add reverse edge to symmetrize the original graph
  k: embedding dimension for spectral methods
  num_hops: number of power iteration for PowerNet/SGC
  '''
  #preprocess data
  nx_graph = to_networkx(data,to_undirected=undirected, remove_self_loops=True)
  if kc is not None: #only use the subgraph with each node has degree > kcore number
    nx_graph = nx.k_core(nx_graph, k=kc) #nx.k_core(nx_graph, k=kc)
    kcoreIDs = list(nx_graph.nodes())
    data.x = data.x[kcoreIDs]
    data.y = data.y[kcoreIDs]
    data.train_mask = data.train_mask[kcoreIDs]
    data.val_mask = data.val_mask[kcoreIDs]
    data.test_mask = data.test_mask[kcoreIDs]

  A = nx.to_numpy_array(nx_graph)
  if self_loops:
    A = A + np.eye(A.shape[0])

  if individual:
    A_deg = A.sum(axis=0)
  A_tensor = torch.FloatTensor(A)
  Xouter = data.x.numpy() @ data.x.numpy().T
  print(f'A: {A.shape}, Cov(X): {Xouter.shape}')
  if power_only == False:
    #spectral methods
    acc_X, acc_ASE, acc_X_ASE, feat_X = run_spectral(data, A, Xouter, k=k, device=device)
    results_spectral = {'Cov(X)': acc_X, 'ASE': acc_ASE, 'Cov(X)_ASE': acc_X_ASE}
  else: 
    evaluesX, evectorsX = eigsh(Xouter, k=k)
    feat_X = torch.FloatTensor(evectorsX[:, :k])

  results, results_last, results_fl = {}, {}, {}
  degs = A_tensor.sum(axis=0).clamp(min=1)
  norm = torch.pow(degs, -0.5)
  Atilde = torch.diag(norm) @ A_tensor @ torch.diag(norm) #[BUG: sparse matrix mult faster]
  #Different power iterations
  Powers_A_norm = power_iterate(A_tensor, torch.FloatTensor(feat_X), K=num_hops, upper_tri=upper_tri)
  Powers_Lap_norm = power_iterate(Atilde, torch.FloatTensor(feat_X), K=num_hops, upper_tri=upper_tri)
  Powers_RW_norm = power_iterate(A_tensor, torch.FloatTensor(feat_X), K=num_hops, upper_tri=upper_tri, rw=True)
  if power_only == False:
    Powers_A =  power_iterate(A_tensor, torch.FloatTensor(feat_X), K=num_hops, upper_tri=upper_tri, normalize=False)
    Powers_Lap = power_iterate(Atilde, torch.FloatTensor(feat_X), K=num_hops, upper_tri=upper_tri, normalize=False) #SGC
    Powers_RW = power_iterate(A_tensor, torch.FloatTensor(feat_X), K=num_hops, upper_tri=upper_tri, rw=True, normalize=False) #SIGN

  #run all
  if power_only:
    names = ['A_norm', 'Lap_norm', 'RW_norm']
    list_powers = [Powers_A_norm, Powers_Lap_norm, Powers_RW_norm]
  else:
    names = ['A_norm', 'A', 'Lap_norm', 'Lap', 'RW_norm', 'RW']
    list_powers = [Powers_A_norm, Powers_A, Powers_Lap_norm, Powers_Lap, Powers_RW_norm, Powers_RW]
  for name, Powers in zip(names, list_powers):
    print(f"running {name}")
    results[name] = run_exp(Powers, data, K=num_hops+1, all_iter='all', individual=individual, eval_every=eval_every, device=device)
    results_last[name] = run_exp(Powers, data, K=num_hops+1, all_iter='last', individual=individual, eval_every=eval_every, device=device)
    results_fl[name] = run_exp(Powers, data, K=num_hops+1, all_iter='first_last', individual=individual, eval_every=eval_every, device=device)

  if power_only:
    return results, results_last, results_fl
  else:
    return results_spectral, results, results_last, results_fl


def train_test_split(data, runs=10):
    #first create a new in-memory dataset, and then add the train/val/test masks
    #same 10 random splits as: https://github.com/cf020031308/3ference/blob/master/main.py
    data_new = Data(x=data.x, edge_index=data.edge_index, y=data.y,
              train_mask=torch.zeros(data.y.size()[0],runs, dtype=torch.bool),
              test_mask=torch.zeros(data.y.size()[0],runs, dtype=torch.bool),
              val_mask=torch.zeros(data.y.size()[0],runs, dtype=torch.bool))
    n_nodes = data.num_nodes
    val_num = test_num = int(n_nodes * 0.2)

    for run in range(runs):
        torch.manual_seed(run)
        idx = torch.randperm(n_nodes)
        data_new.train_mask[idx[(val_num + test_num):], run] = True
        data_new.val_mask[idx[:val_num], run] = True
        data_new.test_mask[idx[val_num:(val_num + test_num)], run] = True
    return data_new

def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    results_all = {} 
    if args.kcore:
      kc = args.kcore_size
      k = 10
    else:
      kc = None
      k = 100

    if args.datasets == 'wiki':
      datafunction = WikipediaNetwork
      names = ['chameleon', 'squirrel']
    elif args.datasets == 'planetoid':
      datafunction = Planetoid
      names = ['Cora', 'CiteSeer']
    elif args.datasets == 'webkb':
      datafunction = WebKB
      names = ['Cornell', 'Texas', 'Wisconsin']
      k = 10
    elif args.datasets == 'amazon':
      datafunction = Amazon
      names = ['computers', 'photo']
    elif args.datasets == 'actor':
      datafunction = Actor
      names = ['Actor']
    elif args.datasets == 'coauthor':
      datafunction = Coauthor
      names = ['CS']
    else:
      raise NotImplementedError
    
    for name in names:
      print(f"dowloading {name}")
      if args.datasets == 'planetoid':
        data_all = datafunction(root=args.data_path, name=name, split='geom-gcn') 
      elif args.datasets == 'actor':
        data_all = datafunction(root=args.data_path)
      else: 
        data_all = datafunction(root=args.data_path, name=name)

      data = data_all[0]
      if args.datasets in ['amazon', 'coauthor']: #create 60/20/20 split for 10 runs
        data = train_test_split(data)
        
      if args.pow_only:
        results, results_last, results_fl = run_data(data, undirected=True, k=k, self_loops=args.loop, kc=kc, 
                    power_only=True, eval_every=args.eval_every, device=device)
        results_all[name] = (results, results_last, results_fl)

      else:
        results_spectral, results, results_last, results_fl = run_data(data, undirected=True, k=k, self_loops=args.loop, kc=kc,
                    power_only=False, eval_every=args.eval_every, device=device)
        results_all[name] = (results_spectral, results, results_last, results_fl)

    file_name = os.path.join(args.result_path, args.datasets + ".pkl")
    if args.kcore:
      file_name = os.path.join(args.result_path, args.datasets + "_kcore.pkl")
    pickle.dump(results_all, open(file_name, "wb" ))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PowerEmbed")
    parser.add_argument("--datasets", type=str, default="wiki", help="datasets: wiki / planetoid / webkb / amazon / actor / coauthor")
    parser.add_argument("--data_path", type=str, default="./dataset/", help="dataset folder path")
    parser.add_argument("--result_path", type=str, default="./result_noUT/", help="dataset folder path")
    parser.add_argument("--R", type=int, default=10,help="number of hops")
    parser.add_argument("--eval_every", type=int, default=1,help="evaluation every k epochs")
    parser.add_argument("--loop", action='store_true', help="add self loop to the graph")
    parser.add_argument("--pow_only", action='store_true', help="only run experiments for PowerEmbed")
    parser.add_argument("--kcore", action='store_true', help="run on kcore subgraph")
    parser.add_argument("--kcore_size", type=int, default=4, help="kcore subgraph degree")

    args = parser.parse_args()

    print(args)
    main(args)