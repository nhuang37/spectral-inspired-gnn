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


class Semi_POWER(torch.nn.Module):
    def __init__(self, g, num_feat, Ys, K, dropout, rw=False, lap=False, device="cuda:0"):
        """
        g: graph, pytorch 2D tensor (n by n)
        num_feat: node feature dimension
        Ys: labelse
        K: number of power iterations
        dropout: dropout probability for SIGN_POWER (classifier head)
        rw: indicator of whether to use the normalized random walk graph operator
        lap: indicator of whether to use the normalized symmetric graph laplacian operator
        device: gpu/cpu location
        """
        super(Semi_POWER, self).__init__()
        self.g = g
        self.K = K
        self.dropout = dropout
        self.rw = rw
        self.lap = lap
        self.device = device
        self.layers = nn.ModuleList()
        for i in range(K):
            self.layers.append(nn.Linear(num_feat, num_feat, bias=False)) #BUG: change to NO BIAS
        #Init weights to be identity matrices
        self.layers.apply(self._init_weights)
        #print(self.layers[0].weight.data)
        in_size = [num_feat] * (K + 1)
        num_hidden = [input_size * 2 for input_size in in_size]
        self.classifer = SIGN_POWER(in_size, num_hidden, len(torch.unique(Ys)), num_hops=K+1,
                                    n_layers=2, dropout=self.dropout, input_drop=0, subnet=True, device=device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            num_feat = m.weight.data.shape[0]
            m.weight.data.copy_(torch.eye(num_feat))

    def forward(self, feat):
        feat = torch.tensor(feat, dtype=torch.float, device=self.device)
        g = torch.tensor(self.g, dtype=torch.float, device=self.device)
        # g = self.g
        powers = [feat]
        if self.rw:
            # Dinv = 1 / self.g.sum(axis=1)
            Dinv = 1 / torch.sum(g, dim=1)
            g = g * Dinv[:, None]  # (500,) --> (500, 1)
        if self.lap:
            degs = torch.sum(g, dim=1).clamp(min=1)
            norm = torch.pow(degs, -0.5)
            g = torch.diag(norm) @ g @ torch.diag(norm) #[BUG: sparse matrix mult faster]
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
        
###BUG: have to perform full batch graph conv, not very scalable compared to unsup-power
def train(model, feats, labels, loss_fcn, optimizer, train_mask):
    model.train()
    loss = loss_fcn(model(feats)[train_mask], labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test(model, feats, labels, train_nid, val_nid, test_nid):
    model.eval()
    preds = torch.argmax(model(feats), dim=-1)
    train_res = (preds[train_nid] == labels[train_nid]).sum()/len(train_nid)
    val_res = (preds[val_nid] == labels[val_nid]).sum()/len(val_nid)
    test_res = (preds[test_nid] == labels[test_nid]).sum()/len(test_nid)
    return train_res, val_res, test_res

def run_model(model, feats, labels,  train_nid, val_nid, test_nid,  
                  lr, weight_decay, num_epochs, eval_every, verbose=False):
    #optim
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    # Start training
    best_epoch = 0
    best_val = 0
    best_test = 0

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train(model, feats, labels, loss_fcn, optimizer, train_nid)

        if epoch % eval_every == 0:
            with torch.no_grad():
                acc = test(model, feats, labels, train_nid, val_nid, test_nid)
            end = time.time()
            log = "Epoch {}, Time(s): {:.4f}, ".format(epoch, end - start)
            log += "Acc: Train {:.4f}, Val {:.4f}, Test {:.4f}".format(*acc)
            if verbose:
              print(log)
            if acc[1] > best_val:
                best_epoch = epoch
                best_val = acc[1]
                best_test = acc[2]

    if verbose:
      print("Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
        best_epoch, best_val, best_test))
    
    return best_val, best_test

def run_exp(data, g, feat_X, K, num_feat, rw_flag, lap_flag, device='cuda:0', eval_every = 1,
            lr=1e-2, n_layer=2, dropout=0.5, verbose=False):
  '''
  data: pytorch geometric data class
  g: graph operator (adjacency/RW/Lap)
  feat_X: node features (or dimensionality-reduced features)
  K: maximum number of power_iteration
  all_iter: 'all' - use all iterations; 'last' - use the last iteration; 'first_last" - use the first (i.e. input feat) and the last iter
  '''
  acc_POWER = {}
  num_classes = len(torch.unique(data.y))
  labels = data.y.to(device)
  num_epochs = 100
  weight_decay = 0
  runs = data.train_mask.shape[1]
  #print('Total runs=', runs)
  #results
  for k in range(1,K+1):
    acc_POWER[k] = []
    
    for run in range(runs):
      train_mask = data.train_mask[:,run]
      val_mask = data.val_mask[:,run]
      test_mask = data.test_mask[:, run]
      nids = torch.arange(data.num_nodes)
      train_nid = nids[train_mask]
      val_nid = nids[val_mask]
      test_nid = nids[test_mask]
      model = Semi_POWER(g, num_feat, data.y, k, dropout, rw=rw_flag, lap=lap_flag, device=device)
      #g, num_feat, Ys, K, rw=False, lap=False, device="cuda:0"
      model = model.to(device)
      best_val, best_test = run_model(model, feat_X, labels, train_nid, val_nid, test_nid,  
            lr, weight_decay, num_epochs, eval_every, verbose=verbose)
      acc_POWER[k].append(best_test.cpu().numpy())
  return acc_POWER

def run_data(data, self_loops=True, undirected=True, k=10, num_hops=10, 
             eval_every=1, dropout=0.5, device='cuda:0'):
  '''
  data: pytorch geometric data
  undirected: If true, add reverse edge to symmetrize the original graph
  k: embedding dimension for spectral methods
  num_hops: number of power iteration for PowerNet/SGC
  '''
  #preprocess data
  nx_graph = to_networkx(data,to_undirected=undirected, remove_self_loops=True)
  A = nx.to_numpy_array(nx_graph)
  if self_loops:
    A = A + np.eye(A.shape[0])

  A_tensor = torch.FloatTensor(A)
  Xouter = data.x.numpy() @ data.x.numpy().T
  print(f'A: {A.shape}, Cov(X): {Xouter.shape}')
  
  evaluesX, evectorsX = eigsh(Xouter, k=k)
  feat_X = torch.FloatTensor(evectorsX[:, :k])
  feat_X = feat_X.to(device)
  labels = data.y.to(device)

  results= {}

  #run all
  name_dict = {'A_norm':(False, False), 'Lap_norm':(False, True), 'RW_norm':(True, False)}
  for name, flag in name_dict.items():
    print(f"running {name}")
    rw_flag, lap_flag = flag
    results[name] = run_exp(data, A_tensor, feat_X, num_hops, k, rw_flag, lap_flag, device=device, eval_every = eval_every,
            lr=1e-2, n_layer=2, dropout=dropout, verbose=False)
    
  return results


def train_test_split(data):
    #first create a new in-memory dataset, and then add the train/val/test masks
    #same split as: https://github.com/cf020031308/3ference/blob/master/main.py
    data_new = Data(x=data.x, edge_index=data.edge_index, y=data.y,
              train_mask=torch.zeros(data.y.size()[0],10, dtype=torch.bool),
              test_mask=torch.zeros(data.y.size()[0],10, dtype=torch.bool),
              val_mask=torch.zeros(data.y.size()[0],10, dtype=torch.bool))
    n_nodes = data.num_nodes
    val_num = test_num = int(n_nodes * 0.2)

    for run in range(10):
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
    #default node feat dimension
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

      results_all[name] = run_data(data, undirected=True, k=k, num_hops=args.R,
                           self_loops=args.loop, eval_every=args.eval_every, dropout=args.drop_prob,
                           device=device)


    file_name = os.path.join(args.result_path, args.datasets + ".pkl")

    pickle.dump(results_all, open(file_name, "wb" ))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PowerEmbed")
    parser.add_argument("--datasets", type=str, default="wiki", help="datasets: wiki / planetoid / webkb / amazon / actor / coauthor")
    parser.add_argument("--data_path", type=str, default="./dataset/", help="dataset folder path")
    parser.add_argument("--result_path", type=str, default="./result_semi/", help="dataset folder path")
    parser.add_argument("--R", type=int, default=10,help="number of hops")
    parser.add_argument("--drop_prob", type=float, default=0.5, help="dropout value")
    parser.add_argument("--eval_every", type=int, default=1,help="evaluation every k epochs")
    parser.add_argument("--loop", action='store_true', help="add self loop to the graph")


    args = parser.parse_args()

    print(args)
    main(args)
