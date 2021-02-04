import numpy as np
import pdb
import sklearn
import time
from src.spectral_cluster_model import (weights_to_graph,
                                        cnn_tensors_to_flat_weights_and_graph,
                                        delete_isolated_ccs_refactored)
from src.utils import load_weights

model_path = "./models/10112019/cifar10_cnn_10epochs/cifar10-cnn-unpruned-weights.pckl"
is_mlp = False
num_components = 4

model_weights = load_weights(model_path)

t1 = time.time()

if is_mlp:
    weights_ = model_weights
    adj_mat_ = weights_to_graph(model_weights)
else:
    weights_, adj_mat_ = cnn_tensors_to_flat_weights_and_graph(
        model_weights, 'one_on_n', True
    )

weights, adj_mat, node_mask, _ = delete_isolated_ccs_refactored(
    weights_, adj_mat_, is_testing=False
)

t2 = time.time()

# here's where the magic happens

embedding = sklearn.manifold.spectral_embedding(adjacency = adj_mat,
                                                n_components = num_components,
                                                eigen_solver = 'amg',
                                                eigen_tol = 1e-5,
                                                drop_first = False)

t3 = time.time()

print("\nTime to create adjacency matrix:", t2 - t1)
print("\nTime to do eigenvalue finding:", t3 - t2)
