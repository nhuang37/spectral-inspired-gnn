# Spectral-Inspired Graph Neural Networks

This repo contains implementations of spectral-inspired GNNs, which present two simple techniques to inject global information in message-passing GNNs, motivated from spectral graph embedding.

## Dependencies
- Python 3.7+
- Pytorch 1.10+
- [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)


You can follow the code below to install pytorch-geometric
```
import os
import torch
os.environ['TORCH'] = torch.__version__
pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```

## PowerEmbed
Run PowerEmbed on the 10 benchmark graphs with the default implementation:
```
python power.py --datasets wiki --loop --data_path YOUR_DATA_PATH --result_path YOUR_RESULT_PATH
```
datasets include options with: wiki / planetoid / webkb / amazon / coauthor

## Graph Decomposition
Run the graph decomposition experiments on subgraphs constrcuted from Cora and CiteSeer:
```
!python graph_decomp.py --datasets Cora
```

## Simulations and visualizations
- All simulations using stochastic block models can be found in ```power_dense_sparse_decomposition.ipynb```
- Our experiment results and ablation studies can be found in ```Plot_results.ipynb```
