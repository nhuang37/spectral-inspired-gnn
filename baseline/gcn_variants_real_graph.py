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
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.nn import GATConv, GCNConv, ChebConv
import dgl
from dgl.nn import GCN2Conv, JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter


# reproducibility
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
    def __init__(self, in_feats, hidden, out_feats, num_hops, n_layers, dropout, input_drop, subnet=True,
                 device="cuda:0"):
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
    def __init__(self, A, num_feat, Ys, layer=1, dropout=0.5, alpha=0.1, device="cuda:0"):
        super(GCNII, self).__init__()
        A = torch.tensor(A, dtype=torch.float, device=device)
        edges = torch.where(A == 1)
        g = dgl.graph(edges)
        self.device = device
        self.g = g
        self.num_feat = num_feat
        self.Ys = Ys
        self.layer = layer
        self.alpha = alpha
        self.embedding = GCN2Conv(num_feat, layer=layer, alpha=alpha)  # , activation=F.relu)
        self.classifier = SIGN_POWER(num_feat, num_feat, len(np.unique(Ys)),
                                     num_hops=None, n_layers=2, dropout=dropout, input_drop=0, subnet=False, device=device)

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


# ref: https://github.com/jianhao2016/GPRGNN
class GPRGNN(torch.nn.Module):
    def __init__(self, A, num_feat, Ys, K=10, alpha=0.1, dropout=0.5,
                 Init='Random', Gamma=None, ppnp='GPR_prop', dprate=0.5, device="cuda:0"):
        super(GPRGNN, self).__init__()
        self.classifier = SIGN_POWER(num_feat, num_feat, len(np.unique(Ys)),
                                     num_hops=None, n_layers=2, dropout=dropout, input_drop=0, subnet=False, device=device)

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


class JKNet(nn.Module):
    def __init__(self, A, num_feat, Ys, n_layers=1, dropout=0.5, device="cuda:0"):
        super(JKNet, self).__init__()
        A = torch.tensor(A, dtype=torch.float, device=device)
        self.A = A
        edges = torch.where(A == 1)
        # g = dgl.graph(edges)
        self.device = device
        # self.g = g
        self.num_feat = num_feat
        self.Ys = Ys

        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(GCNConv(num_feat, num_feat))

        self.jk = JumpingKnowledge(mode='cat')
        in_size = num_feat * (n_layers + 1)
        num_hidden = in_size * 2
        self.classifier = SIGN_POWER(in_size, num_hidden, len(np.unique(Ys)),
                                     num_hops=None, n_layers=2, dropout=dropout, input_drop=0, subnet=False,
                                     device=device)
        # self.classifier = SIGN_POWER(in_size, num_hidden, len(np.unique(Ys)), num_hops=n_layers + 1,
        #                             n_layers=2, dropout=dropout, input_drop=0, subnet=True, device=device)

    def forward(self, feat):
        feat = torch.tensor(feat, dtype=torch.float, device=self.device)
        edge_index = torch.stack(torch.where(self.A))
        emb = [feat]
        for layer in self.layers:
            feat = layer(feat, edge_index)
            emb.append(feat)
        out = self.classifier(self.jk(emb))
        return out


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
    train_res = (preds[train_nid] == labels[train_nid]).sum() / len(train_nid)
    val_res = (preds[val_nid] == labels[val_nid]).sum() / len(val_nid)
    test_res = (preds[test_nid] == labels[test_nid]).sum() / len(test_nid)
    return train_res, val_res, test_res


def run_model(model, feats, labels, train_nid, val_nid, test_nid,
              lr, weight_decay, num_epochs, eval_every, verbose=False):
    # optim
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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


def run_gcn_variant(data, model_name, device='cuda:0', eval_every=1, self_loops=True, undirected=True,
                    lr=1e-2, n_layer=2, dropout=0.5, verbose=False, k=10):
    '''
    data: pytorch geometric data class
    k: embedding dimension for spectral methods, i.e. num_feat
    model_name: GCNII or GPRGNN
    '''
    # preprocess data
    nx_graph = to_networkx(data, to_undirected=undirected, remove_self_loops=True)
    A = nx.to_numpy_array(nx_graph)
    if self_loops:
        A = A + np.eye(A.shape[0])

    num_classes = len(torch.unique(data.y))
    labels = data.y.to(device)
    num_feat = data.x.shape[1]

    Xouter = data.x.numpy() @ data.x.numpy().T
    print(f'A: {A.shape}, Cov(X): {Xouter.shape}')
    evaluesX, evectorsX = eigsh(Xouter, k=k)
    feat_X = torch.FloatTensor(evectorsX[:, :k])
    feat_X = feat_X.to(device)

    num_epochs = 100
    weight_decay = 0
    runs = data.train_mask.shape[1]  # 10
    # print('Total runs=', runs)
    # results
    acc_variant = []
    for run in range(runs):
        print(run)
        train_mask = data.train_mask[:, run]
        val_mask = data.val_mask[:, run]
        test_mask = data.test_mask[:, run]
        nids = torch.arange(data.num_nodes)
        train_nid = nids[train_mask]
        val_nid = nids[val_mask]
        test_nid = nids[test_mask]
        if model_name == "GCNII":
            model = GCNII(A, num_feat, data.y, layer=10, dropout=dropout, device=device)
        elif model_name == "GPRGNN":
            model = GPRGNN(A, num_feat, data.y, K=10, dropout=dropout, device=device)
        elif model_name == "JKNet":
            num_feat = k
            model = JKNet(A, num_feat, data.y, n_layers=10, dropout=dropout, device=device)
        else:
            raise NotImplementedError
        model = model.to(device)
        if model_name == "JKNet":  # concatenation will cause OOM --> use top-k eigenvectors of Cov(data.x)
            best_val, best_test = run_model(model, feat_X, labels, train_nid, val_nid, test_nid,
                                            lr, weight_decay, num_epochs, eval_every, verbose=verbose)
        else:
            best_val, best_test = run_model(model, data.x, labels, train_nid, val_nid, test_nid,
                                            lr, weight_decay, num_epochs, eval_every, verbose=verbose)
        acc_variant.append(best_test.cpu().numpy())
    return acc_variant


def train_test_split(data):
    # first create a new in-memory dataset, and then add the train/val/test masks
    # same split as: https://github.com/cf020031308/3ference/blob/master/main.py
    data_new = Data(x=data.x, edge_index=data.edge_index, y=data.y,
                    train_mask=torch.zeros(data.y.size()[0], 10, dtype=torch.bool),
                    test_mask=torch.zeros(data.y.size()[0], 10, dtype=torch.bool),
                    val_mask=torch.zeros(data.y.size()[0], 10, dtype=torch.bool))
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
    # if args.model == "GCNII":
    #     device = torch.device("cpu")

    results_all = {}
    # default node feat dimension
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
        if args.datasets in ['amazon', 'coauthor']:  # create 60/20/20 split for 10 runs
            data = train_test_split(data)
        if args.model == "GCNII" or args.model == "GPRGNN" or args.model == "JKNet":
            results_all[name] = run_gcn_variant(data, args.model, self_loops=args.loop, eval_every=args.eval_every,
                                                dropout=args.drop_prob, undirected=True, device=device, k=k)
        else:
            raise NotImplementedError

    file_name = os.path.join(args.result_path, args.datasets + ".pkl")

    pickle.dump(results_all, open(file_name, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PowerEmbed")
    parser.add_argument("--datasets", type=str, default="wiki",
                        help="datasets: wiki / planetoid / webkb / amazon / actor / coauthor")
    parser.add_argument("--data_path", type=str, default="./dataset/", help="dataset folder path")
    parser.add_argument("--result_path", type=str, default="./result_gcnii/", help="dataset folder path")
    parser.add_argument("--R", type=int, default=10, help="number of hops")
    parser.add_argument("--drop_prob", type=float, default=0.5, help="dropout value")
    parser.add_argument("--eval_every", type=int, default=1, help="evaluation every k epochs")
    parser.add_argument("--loop", action='store_true', help="add self loop to the graph")
    parser.add_argument("--model", type=str, default="GCNII", help="model: GCNII / GPRGNN")

    args = parser.parse_args()

    print(args)
    main(args)