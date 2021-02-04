from math import isclose
from collections import Counter

import numpy as np
import networkx as nx
import pytest

from src.spectral_cluster_model import (clustering_experiment,
                                        weights_to_graph, compute_ncut,
                                        shuffle_weights,
                                        shuffle_weights_nonzero,
                                        shuffle_weights_nonzero_distribution,
                                        shuffle_weights_layer_all_distribution)
from src.experiment_tagging import get_model_path
from src.pointers import DATA_PATHS
from src.utils import get_weights_paths, load_model2, preprocess_dataset
from src.cnn.convertor import cnn2mlp

import tensorflow as tf


# The path to the models directory, given that we'll run this file from
# the project directory with
# `pytest src/tests/test_spectral_clustering.py'
BASE_PATH = './models/'


def test_compute_ncut(epsilon=1e-8):
    weights_small = [np.array([[12, -1.3],
                               [0, 7.4]]),
                     np.array([[-8.9, 0.15],
                               [0.23, -5.7]])]
    adj_mat_small = weights_to_graph(weights_small).toarray()
    clustering_small = [0, 1, 0, 1, 0, 1]

    assert isclose(compute_ncut(adj_mat_small, clustering_small, epsilon),
                   0.09895, abs_tol=1e-3)


    weights_big = [np.array([[5.4, 0.03, 0, 0],
                             [0, 0, -0.4, 5.5],
                             [-12, 0, 4.8, -0.07],
                             [0, 6.3, 0.001, 7]]),
                   np.array([[-13.3, -0.4, 6, 0.1],
                             [0.7, -8, 0, -5.8],
                             [7.4, 0.3, -15, 0],
                             [0, 0, 0, -12.3]])]
    adj_mat_big = weights_to_graph(weights_big).toarray()
    clustering_big = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    assert isclose(compute_ncut(adj_mat_big, clustering_big, epsilon),
                   0.0372, abs_tol=1e-3)


    # test on a random graph, and use networkx implementation
    G = nx.generators.random_graphs.fast_gnp_random_graph(20, 0.8)
    adj_mat = nx.adj_matrix(G)

    assert isclose(nx.algorithms.cuts.normalized_cut_size(G,
                                                          list(G.nodes)[::2],
                                                          list(G.nodes)[1::2]),
             compute_ncut(adj_mat, [0, 1] * 10, 0))


@pytest.mark.skip(reason="not testing CNN right now")
def test_two_methods_cnn_clustering_pvalue():
    
    # See comment above about BASE_PATH
    config_updates = {'weights_path': get_weights_paths(get_model_path('CNN-MNIST',
                                                        model_base_path=BASE_PATH))[True],
                      'with_labels': False,
                      'with_shuffle': False,
                      'num_clusters': 4,
                      'is_testing': True}

    experiment_run = clustering_experiment.run(config_updates=config_updates,
                                               named_configs=['cnn_config'])


@pytest.mark.skip(reason="not testing CNN right now")
@pytest.mark.parametrize('dataset', ['mnist', 'fashion'])  # 'line', 'cifar10'
def test_cnn2mlp(dataset, is_unpruned=True, n_datapoints=100, abs_tol=0.075):

    ds = preprocess_dataset(DATA_PATHS[dataset])
    X, y = ds['X_test'][:n_datapoints], ds['y_test'][:n_datapoints]

    type_ = 'unpruned' if is_unpruned else 'pruned'

    # See comment above about BASE_PATH
    model_path = (get_model_path(f'CNN-{dataset}'.upper(), model_base_path=BASE_PATH)
                  / f'{dataset}-cnn-{type_}.h5'.lower())
    cnn_model = load_model2(model_path)
    mlp_model = cnn2mlp(model_path, verbose=True)

    y_pred = mlp_model.predict_classes(X)

    acc_mlp = np.mean(y == y_pred)

    _,  acc_cnn = cnn_model.evaluate(X.reshape(-1, 28, 28, 1),
                                     tf.keras.utils.to_categorical(y))

    print('CNN', acc_cnn, 'MLP', acc_mlp)

    assert isclose(acc_cnn, acc_mlp, abs_tol=abs_tol)


def test_shuffle_methods():

    weights = np.dstack([np.arange(10) + 1,
                         np.zeros(10)]).reshape(4, 5)

    ###

    shuffled = shuffle_weights(weights)

    assert weights.shape == shuffled.shape
    assert Counter(weights.flatten()) == Counter(shuffled.flatten())
    assert all((x != y).any() for x, y in zip(*(np.nonzero(weights), np.nonzero(shuffled))))

    ###

    nonzero_shuffled = shuffle_weights_nonzero(weights)

    assert weights.shape == nonzero_shuffled.shape
    assert Counter(weights.flatten()) == Counter(nonzero_shuffled.flatten())
    assert all((x == y).all() for x, y in zip(*(np.nonzero(weights), np.nonzero(nonzero_shuffled))))

    ###

    nonzero_distribution_shuffled = shuffle_weights_nonzero_distribution(weights)

    assert weights.shape == nonzero_distribution_shuffled.shape
    assert Counter(weights.flatten()) != Counter(nonzero_distribution_shuffled.flatten())
    assert all((x == y).all() for x, y in zip(*(np.nonzero(weights), np.nonzero(nonzero_distribution_shuffled))))
    assert (weights[weights !=0].mean()
            != nonzero_distribution_shuffled[nonzero_distribution_shuffled != 0].mean())


    ###

    all_distribution_shuffled = shuffle_weights_layer_all_distribution(weights)

    assert weights.shape == all_distribution_shuffled.shape
    assert Counter(weights.flatten()) != Counter(all_distribution_shuffled.flatten())
    assert all((x != y).any() for x, y in zip(*(np.nonzero(weights), np.nonzero(all_distribution_shuffled))))
    assert (weights[weights !=0].mean()
            != all_distribution_shuffled[all_distribution_shuffled != 0].mean())
